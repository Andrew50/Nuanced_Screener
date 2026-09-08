"""Offline multilabel evaluation from explicit reviews. Not a backtester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .types import CandidateResult, PreparedScan, ReviewRecord

_BINARY_VERDICTS = frozenset({"match", "no_match"})


@dataclass(frozen=True)
class SetupConfusion:
    setup_id: str
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    labeled: int
    precision: float | None
    recall: float | None


@dataclass(frozen=True)
class EvaluationReport:
    run_id: str
    eligible_pairs: int
    labeled_pairs: int
    excluded_unsure: int
    excluded_unreviewed: int
    excluded_prompt_examples: int
    excluded_uncertain_verdicts: int
    coverage: float | None
    by_setup: tuple[SetupConfusion, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "eligible_pairs": self.eligible_pairs,
            "labeled_pairs": self.labeled_pairs,
            "excluded_unsure": self.excluded_unsure,
            "excluded_unreviewed": self.excluded_unreviewed,
            "excluded_prompt_examples": self.excluded_prompt_examples,
            "excluded_uncertain_verdicts": self.excluded_uncertain_verdicts,
            "coverage": self.coverage,
            "by_setup": [
                {
                    "setup_id": row.setup_id,
                    "true_positive": row.true_positive,
                    "false_positive": row.false_positive,
                    "false_negative": row.false_negative,
                    "true_negative": row.true_negative,
                    "labeled": row.labeled,
                    "precision": row.precision,
                    "recall": row.recall,
                }
                for row in self.by_setup
            ],
            "notes": list(self.notes),
        }


def evaluate_run(
    *,
    prepared: PreparedScan,
    results: Sequence[CandidateResult],
    reviews: Sequence[ReviewRecord],
    run_id: str,
) -> EvaluationReport:
    """Evaluate explicit reviewed candidate/setup pairs.

    Unsure and unreviewed pairs are excluded. Prompt examples are not held-out cases.
    A rejected setup is not a negative for every other setup.
    """

    example_keys = {
        (str(ex.ticker).strip().upper(), ex.asof_date)
        for ex in prepared.examples
        if ex.ticker and ex.asof_date is not None
    }
    by_id = {row.candidate_id: row for row in results}
    current = _current_reviews(reviews)

    eligible_pairs = 0
    labeled = 0
    unsure = 0
    unreviewed = 0
    prompt_excluded = 0
    uncertain_verdicts = 0
    counts: dict[str, list[int]] = {}

    for cand in prepared.candidates:
        result = by_id.get(cand.candidate_id)
        assessments = {a.setup_id: a for a in (result.assessments if result else ())}
        prompt_hit = (cand.ticker, cand.asof_date) in example_keys
        for setup_id in cand.eligible_setup_ids:
            eligible_pairs += 1
            if prompt_hit:
                prompt_excluded += 1
                continue
            review = current.get((cand.candidate_id, setup_id))
            if review is None:
                unreviewed += 1
                continue
            if review.judgment == "unsure":
                unsure += 1
                continue
            assessment = assessments.get(setup_id)
            if assessment is None or assessment.verdict not in _BINARY_VERDICTS:
                uncertain_verdicts += 1
                continue
            pred_pos = assessment.verdict == "match"
            if review.judgment == "agree":
                truth_pos = pred_pos
            elif review.judgment == "disagree":
                truth_pos = not pred_pos
            else:
                unsure += 1
                continue
            bucket = counts.setdefault(setup_id, [0, 0, 0, 0])
            if truth_pos and pred_pos:
                bucket[0] += 1
            elif (not truth_pos) and pred_pos:
                bucket[1] += 1
            elif truth_pos and (not pred_pos):
                bucket[2] += 1
            else:
                bucket[3] += 1
            labeled += 1

    by_setup = []
    for setup_id, (tp, fp, fn, tn) in sorted(counts.items()):
        pos_pred = tp + fp
        pos_truth = tp + fn
        by_setup.append(
            SetupConfusion(
                setup_id=setup_id,
                true_positive=tp,
                false_positive=fp,
                false_negative=fn,
                true_negative=tn,
                labeled=tp + fp + fn + tn,
                precision=(tp / pos_pred) if pos_pred else None,
                recall=(tp / pos_truth) if pos_truth else None,
            )
        )
    coverage = (labeled / eligible_pairs) if eligible_pairs else None
    notes = (
        "Unsure and unreviewed pairs are excluded from confusion counts.",
        "A rejected setup is not a negative for every other setup.",
        "Reviewing only match flags cannot establish whole-universe recall.",
        "Prompt examples are excluded and are not held-out evaluation cases.",
        "Precision/recall are None when the denominator is zero.",
    )
    return EvaluationReport(
        run_id=run_id,
        eligible_pairs=eligible_pairs,
        labeled_pairs=labeled,
        excluded_unsure=unsure,
        excluded_unreviewed=unreviewed,
        excluded_prompt_examples=prompt_excluded,
        excluded_uncertain_verdicts=uncertain_verdicts,
        coverage=coverage,
        by_setup=tuple(by_setup),
        notes=notes,
    )


def evaluate_from_stores(run_id: str, *, store, reviews) -> EvaluationReport:
    prepared = store.load_frozen_inputs(run_id)
    results = store.list_candidate_results(run_id)
    records = list(reviews.list_run_reviews(run_id)) if hasattr(reviews, "list_run_reviews") else []
    if not records:
        for cand in prepared.candidates:
            records.extend(reviews.list_reviews(run_id, cand.candidate_id))
    return evaluate_run(prepared=prepared, results=results, reviews=records, run_id=run_id)


def _current_reviews(reviews: Sequence[ReviewRecord]) -> dict[tuple[str, str], ReviewRecord]:
    current: dict[tuple[str, str], ReviewRecord] = {}
    for rec in reviews:
        if rec.setup_id is None:
            continue
        key = (rec.candidate_id, rec.setup_id)
        prev = current.get(key)
        if prev is None or rec.created_at >= prev.created_at:
            current[key] = rec
    return current

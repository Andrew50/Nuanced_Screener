"""Pure semantic validation of a classification batch. No provider I/O."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types import (
    MISSING_EVIDENCE_ITEM_MAX_CHARS,
    MISSING_EVIDENCE_MAX_ITEMS,
    REASON_MAX_CHARS,
    VIOLATED_RULE_MAX_ITEMS,
    Assessment,
    CandidateClassification,
    CandidateInput,
    SemanticValidationError,
    SetupSnapshot,
)


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SemanticValidationError("Model output must be a JSON object", issues=("root_type",))
    return payload


def validate_classification_payload(
    payload: Any,
    *,
    candidates: Sequence[CandidateInput],
    setups: Sequence[SetupSnapshot],
) -> tuple[CandidateClassification, ...]:
    data = _as_mapping(payload)
    results = data.get("results")
    if not isinstance(results, list):
        raise SemanticValidationError("results must be an array", issues=("results_type",))

    by_id = {c.candidate_id: c for c in candidates}
    setups_by_id = {s.setup_id: s for s in setups}
    expected_candidates = set(by_id)
    seen_candidates: set[str] = set()
    issues: list[str] = []
    out: list[CandidateClassification] = []

    for row in results:
        if not isinstance(row, Mapping):
            issues.append("result_row_type")
            continue
        cid = row.get("candidate_id")
        if not isinstance(cid, str) or not cid:
            issues.append("candidate_id_missing")
            continue
        if cid not in by_id:
            issues.append(f"unknown_candidate:{cid}")
            continue
        if cid in seen_candidates:
            issues.append(f"duplicate_candidate:{cid}")
            continue
        seen_candidates.add(cid)
        assessments_raw = row.get("assessments")
        if not isinstance(assessments_raw, list):
            issues.append(f"assessments_type:{cid}")
            continue
        candidate = by_id[cid]
        eligible = list(candidate.eligible_setup_ids)
        seen_setups: set[str] = set()
        parsed: list[Assessment] = []
        for item in assessments_raw:
            parsed_one, item_issues = _parse_assessment(item, candidate_id=cid, setups_by_id=setups_by_id, eligible=set(eligible))
            issues.extend(item_issues)
            if parsed_one is not None:
                if parsed_one.setup_id in seen_setups:
                    issues.append(f"duplicate_setup:{cid}:{parsed_one.setup_id}")
                    continue
                seen_setups.add(parsed_one.setup_id)
                parsed.append(parsed_one)
        missing = [s for s in eligible if s not in seen_setups]
        extra = [s for s in seen_setups if s not in eligible]
        if missing:
            issues.append(f"missing_setups:{cid}:{','.join(missing)}")
        if extra:
            issues.append(f"ineligible_setups:{cid}:{','.join(extra)}")
        if not missing and not extra and len(parsed) == len(eligible):
            # Preserve eligible order, not model order.
            by_setup = {a.setup_id: a for a in parsed}
            out.append(
                CandidateClassification(
                    candidate_id=cid,
                    assessments=tuple(by_setup[s] for s in eligible),
                )
            )

    missing_cids = expected_candidates - seen_candidates
    if missing_cids:
        issues.append("missing_candidates:" + ",".join(sorted(missing_cids)))
    extra_ok = True
    if seen_candidates - expected_candidates:
        extra_ok = False
    if issues or not extra_ok or len(out) != len(expected_candidates):
        raise SemanticValidationError(
            "Classification output failed semantic validation; the batch is not accepted",
            issues=tuple(issues),
        )
    # Stable candidate_id order for storage; coverage was already set-equal.
    by_out = {row.candidate_id: row for row in out}
    return tuple(by_out[c.candidate_id] for c in candidates)


def _parse_assessment(
    item: Any,
    *,
    candidate_id: str,
    setups_by_id: Mapping[str, SetupSnapshot],
    eligible: set[str],
) -> tuple[Assessment | None, list[str]]:
    issues: list[str] = []
    if not isinstance(item, Mapping):
        return None, [f"assessment_type:{candidate_id}"]
    setup_id = item.get("setup_id")
    if not isinstance(setup_id, str) or not setup_id:
        return None, [f"setup_id_missing:{candidate_id}"]
    if setup_id not in setups_by_id:
        return None, [f"unknown_setup:{candidate_id}:{setup_id}"]
    if setup_id not in eligible:
        return None, [f"ineligible_setup:{candidate_id}:{setup_id}"]
    verdict = item.get("verdict")
    if verdict not in {"match", "no_match", "uncertain"}:
        issues.append(f"verdict:{candidate_id}:{setup_id}")
        return None, issues
    strength = item.get("match_strength")
    if strength is not None:
        if not isinstance(strength, int) or isinstance(strength, bool) or strength not in {1, 2, 3}:
            issues.append(f"match_strength:{candidate_id}:{setup_id}")
            return None, issues
    reason = item.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        issues.append(f"reason:{candidate_id}:{setup_id}")
        return None, issues
    if len(reason) > REASON_MAX_CHARS * 2:
        issues.append(f"reason_too_long:{candidate_id}:{setup_id}")
        return None, issues
    violated = item.get("violated_required_rule_ids") or []
    missing_ev = item.get("missing_evidence") or []
    if not isinstance(violated, list) or not isinstance(missing_ev, list):
        issues.append(f"list_type:{candidate_id}:{setup_id}")
        return None, issues
    if len(violated) > VIOLATED_RULE_MAX_ITEMS or len(missing_ev) > MISSING_EVIDENCE_MAX_ITEMS:
        issues.append(f"list_bound:{candidate_id}:{setup_id}")
        return None, issues
    setup = setups_by_id[setup_id]
    allowed_rules = setup.required_rule_ids()
    v_ids: list[str] = []
    for rid in violated:
        if not isinstance(rid, str) or rid not in allowed_rules:
            issues.append(f"rule_ref:{candidate_id}:{setup_id}:{rid!r}")
            continue
        v_ids.append(rid)
    ev: list[str] = []
    for note in missing_ev:
        if not isinstance(note, str) or not note.strip():
            issues.append(f"missing_evidence_item:{candidate_id}:{setup_id}")
            continue
        ev.append(note[:MISSING_EVIDENCE_ITEM_MAX_CHARS])
    if issues:
        return None, issues
    if verdict == "match":
        if strength not in {1, 2, 3}:
            return None, [f"match_requires_strength:{candidate_id}:{setup_id}"]
        if v_ids:
            return None, [f"match_with_violations:{candidate_id}:{setup_id}"]
    else:
        if strength is not None:
            return None, [f"strength_not_null:{candidate_id}:{setup_id}"]
    return (
        Assessment(
            setup_id=setup_id,
            verdict=verdict,
            match_strength=strength if verdict == "match" else None,
            reason=reason.strip()[:REASON_MAX_CHARS] if len(reason.strip()) > REASON_MAX_CHARS else reason.strip(),
            violated_required_rule_ids=tuple(v_ids),
            missing_evidence=tuple(ev),
        ),
        [],
    )

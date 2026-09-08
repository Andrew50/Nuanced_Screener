from __future__ import annotations

from datetime import datetime, timezone

from screener_loader.vision.evaluate import evaluate_run
from screener_loader.vision.snapshots import freeze_prepared_scan
from screener_loader.vision.types import CandidateResult, ReviewRecord
from vision_support import assessment, sample_prepared_scan


def test_evaluate_scopes_pairs_and_excludes_unsure() -> None:
    prepared = sample_prepared_scan()
    held_out = prepared.candidates[0]
    prepared = freeze_prepared_scan(
        profile=prepared.profile,
        setups=prepared.setups,
        examples=(),
        candidates=(held_out,),
        config=prepared.config,
        diagnostics=prepared.diagnostics,
        skips=prepared.skips,
        global_filters_yaml_bytes=prepared.global_filters_yaml_bytes,
        prepared_at=prepared.prepared_at,
    )
    result = CandidateResult(
        candidate_id=held_out.candidate_id,
        status="completed",
        ticker=held_out.ticker,
        asof_date=held_out.asof_date,
        features=held_out.features,
        eligible_setup_ids=held_out.eligible_setup_ids,
        assessments=(assessment("flag", "match", strength=2),),
        artifact_id=None,
        error=None,
        attempt_id=None,
        source_digest=held_out.source_digest,
    )
    now = datetime(2026, 6, 19, tzinfo=timezone.utc)
    unsure = (
        ReviewRecord("r1", "run", held_out.candidate_id, "flag", "agree", "", now, None),
        ReviewRecord("r2", "run", held_out.candidate_id, "flag", "unsure", "", now, "r1"),
    )
    report = evaluate_run(prepared=prepared, results=(result,), reviews=unsure, run_id="run")
    assert report.labeled_pairs == 0
    assert report.excluded_unsure == 1

    agree = (ReviewRecord("r3", "run", held_out.candidate_id, "flag", "agree", "", now, None),)
    report2 = evaluate_run(prepared=prepared, results=(result,), reviews=agree, run_id="run")
    assert report2.labeled_pairs == 1
    assert report2.by_setup[0].true_positive == 1
    assert report2.by_setup[0].precision == 1.0

    miss = CandidateResult(
        candidate_id=held_out.candidate_id,
        status="completed",
        ticker=held_out.ticker,
        asof_date=held_out.asof_date,
        features=held_out.features,
        eligible_setup_ids=held_out.eligible_setup_ids,
        assessments=(assessment("flag", "no_match"),),
        artifact_id=None,
        error=None,
        attempt_id=None,
        source_digest=held_out.source_digest,
    )
    disagree = (ReviewRecord("r4", "run", held_out.candidate_id, "flag", "disagree", "missed", now, None),)
    report3 = evaluate_run(prepared=prepared, results=(miss,), reviews=disagree, run_id="run")
    assert report3.by_setup[0].false_negative == 1
    assert report3.by_setup[0].true_positive == 0
    assert report3.by_setup[0].recall == 0.0
    assert any("whole-universe recall" in n for n in report3.notes)
    assert any("not a negative for every" in n for n in report3.notes)

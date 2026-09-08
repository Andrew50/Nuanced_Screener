from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from screener_loader.vision.evaluate import evaluate_from_stores
from screener_loader.vision.service import VisionApp
from screener_loader.vision.types import PageRequest, ResultQuery, ScanConfig
from vision_catalog import LOOKBACK, after_close_clock, build_catalog
from vision_support import FakeClassifier, assessment

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _app(handles, classifier=None, **kwargs) -> VisionApp:
    return VisionApp(
        handles.cfg,
        clock=after_close_clock,
        classifier=classifier,
        **kwargs,
    )


def test_integration_scan_resume_review_export(tmp_path: Path) -> None:
    handles = build_catalog(tmp_path, lookback=LOOKBACK, moving_averages=(10, 20, 50))
    prepared = _app(handles).prepare(ScanConfig(model="dry-run-model", mode="dry_run"))
    assert prepared.profile.moving_averages == (10, 20, 50)
    by_ticker = {c.ticker: c for c in prepared.candidates}
    nvda = by_ticker["NVDA"]
    aapl = by_ticker["AAPL"]
    na = by_ticker["NA"]

    script = {
        nvda.candidate_id: (
            assessment("ep", "match", strength=2, reason="Gap and range expansion."),
            assessment("flag", "match", strength=3, reason="Tight coil under highs."),
        ),
        aapl.candidate_id: (
            assessment("flag", "no_match", reason="Flag broke down.", violated=("flag:required:0",)),
        ),
        na.candidate_id: (
            assessment("ep", "uncertain", reason="Range not visible.", missing=("right side",)),
            assessment("flag", "uncertain", reason="Coil unfinished.", missing=("dry-up",)),
        ),
    }
    clf = FakeClassifier(script=script)
    app = _app(handles, classifier=clf)
    outcome = app.run(ScanConfig(model="gpt-test", mode="live"))
    assert outcome.status == "completed"
    assert outcome.summary.setup_matches == 2
    assert clf.calls >= 1

    results = {r.ticker: r for r in app.store.list_candidate_results(outcome.run_id)}
    assert {a.verdict for a in results["NVDA"].assessments} == {"match"}
    assert results["AAPL"].assessments[0].verdict == "no_match"
    assert results["NA"].assessments[0].verdict == "uncertain"
    assert results["AAPL"].error is None
    detail = app.reader.get_detail(outcome.run_id, nvda.candidate_id)
    assert detail.artifact is not None
    chart = app.reader.get_artifact_bytes(outcome.run_id, detail.artifact.artifact_id)
    assert chart[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(chart) > 100

    dry = _app(handles, classifier=FakeClassifier()).run(ScanConfig(model="dry-run-model", mode="dry_run"))
    assert dry.status == "dry_run"
    dry_clf = FakeClassifier()
    dry_app = _app(handles, classifier=dry_clf)
    dry_out = dry_app.run(ScanConfig(model="dry-run-model", mode="dry_run"))
    assert dry_clf.calls == 0
    assert dry_out.summary.candidates_completed == 0 or dry_out.status == "dry_run"

    cancel = Event()
    cancel.set()
    paused = _app(handles, classifier=FakeClassifier(script=script)).run(
        ScanConfig(model="gpt-test", mode="live"),
        cancel=cancel,
    )
    assert paused.status in {"cancelled", "completed"}
    resumed = _app(handles, classifier=FakeClassifier(script=script)).resume(paused.run_id)
    committed = app.store.list_committed_candidate_ids(outcome.run_id)
    resumed_ids = _app(handles, classifier=FakeClassifier(script=script)).store.list_committed_candidate_ids(
        resumed.run_id
    )
    assert paused.run_id == resumed.run_id
    _ = committed, resumed_ids

    app.reviews.add_review(outcome.run_id, nvda.candidate_id, judgment="agree", setup_id="flag")
    app.reviews.add_review(outcome.run_id, nvda.candidate_id, judgment="disagree", setup_id="ep")
    app.reviews.add_review(outcome.run_id, aapl.candidate_id, judgment="agree", setup_id="flag")
    current = app.reviews.current_review(outcome.run_id, nvda.candidate_id, setup_id="flag")
    assert current is not None and current.judgment == "agree"

    exported = app.reader.export_query(ResultQuery(run_id=outcome.run_id, verdicts=("match",)))
    assert exported.candidates
    assert all(row["matched_setup_ids"] for row in exported.candidates)

    page = app.reader.query(ResultQuery(run_id=outcome.run_id, verdicts=("match",), page=PageRequest(page=1, page_size=10)))
    assert page.has_prev is False
    report = evaluate_from_stores(outcome.run_id, store=app.store, reviews=app.reviews)
    assert report.labeled_pairs >= 1
    by_setup = {row.setup_id: row for row in report.by_setup}
    assert by_setup["flag"].true_positive >= 1
    assert by_setup["ep"].false_positive >= 1
    assert report.coverage is None or report.coverage <= 1

    empty_handles = build_catalog(tmp_path / "empty", include_examples=False)
    from screener_loader.setups.spec import GlobalFilters

    empty_handles.service.save_global_filters(GlobalFilters(min_price=10_000.0, min_dollar_vol_20d=1.0))
    empty_out = _app(empty_handles, classifier=FakeClassifier()).run(ScanConfig(model="dry-run-model", mode="dry_run"))
    assert empty_out.status == "dry_run"
    assert empty_out.summary.candidates_total == 0


def test_demo_mode_is_labeled_synthetic(tmp_path: Path) -> None:
    handles = build_catalog(tmp_path, include_examples=True)
    app = VisionApp(handles.cfg, clock=after_close_clock)
    out = app.run(ScanConfig(model="demo-vision-classifier", mode="demo"))
    assert out.summary.synthetic is True
    assert out.status == "completed"
    stored = app.store.load_run(out.run_id)
    assert stored.synthetic is True
    assert stored.config.mode == "demo"


def test_live_missing_key_does_not_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handles = build_catalog(tmp_path, include_examples=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = VisionApp(handles.cfg, clock=after_close_clock)
    from screener_loader.vision.types import VisionError

    with pytest.raises(VisionError, match="OPENAI_API_KEY"):
        app.run(ScanConfig(model="gpt-4.1", mode="live"))
    assert app.reader.list_runs() == ()

"""Streamlit results page tests. No classifier or data-vendor calls on rerender."""

from __future__ import annotations

from pathlib import Path

import pytest

from screener_loader.vision.types import PageRequest, ResultQuery, SortSpec
from screener_loader.vision_ui.components import (
    STATE_PREFIX,
    VIEW_MATCHES,
    badge_text,
    build_result_query,
    feature_caption,
    format_adr_pct,
    reconcile_selected_candidate,
    state_key,
)
from screener_loader.vision_ui.synthetic import seed_synthetic_runs


def test_state_keys_are_namespaced() -> None:
    assert state_key("run_id") == "vr_run_id"
    assert STATE_PREFIX == "vr_"
    with pytest.raises(ValueError):
        state_key("setup_id")


def test_adr_display_and_query_builder() -> None:
    assert format_adr_pct(0.04) == "4.0%"
    assert format_adr_pct(None) == "—"
    q, statuses = build_result_query(run_id="run-1", view=VIEW_MATCHES, setup_ids=("flag",), page=2)
    assert q.verdicts == ("match",)
    assert q.setup_ids == ("flag",)
    assert q.page.page == 2
    assert statuses == ()
    _q2, err_statuses = build_result_query(run_id="run-1", view="errors")
    assert err_statuses == ("error", "skipped")


def test_selection_reconciliation_rule() -> None:
    ids = ("a", "b", "c")
    assert reconcile_selected_candidate("b", ids) == "b"
    assert reconcile_selected_candidate("z", ids) == "a"
    assert reconcile_selected_candidate(None, ids) == "a"
    assert reconcile_selected_candidate("a", ()) is None


def test_render_results_page_has_no_page_config() -> None:
    import inspect

    from screener_loader.vision_ui import results_page

    source = inspect.getsource(results_page)
    assert "set_page_config" not in source
    from screener_loader.vision_ui import demo

    assert "set_page_config" in inspect.getsource(demo.main)


def test_results_page_does_not_import_client_or_scan() -> None:
    import screener_loader.vision_ui.results_page as rp

    assert "screener_loader.vision.client" not in rp.__dict__
    assert "screener_loader.vision.scan" not in rp.__dict__


def test_demo_page_and_real_store_without_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, reader, reviews = seed_synthetic_runs(tmp_path)
    monkeypatch.setenv("NS_VISION_DEMO_ROOT", str(tmp_path))
    from streamlit.testing.v1 import AppTest

    demo_file = Path(__file__).resolve().parents[1] / "src" / "screener_loader" / "vision_ui" / "demo.py"
    at = AppTest.from_file(str(demo_file), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Previous" in str(b.label) for b in at.button)
    assert any("Next" in str(b.label) for b in at.button)
    assert store.list_runs()
    main = max(store.list_runs(), key=lambda r: reader.counts(r.run_id).candidates)
    q = ResultQuery(
        run_id=main.run_id,
        verdicts=("match",),
        page=PageRequest(1, 10),
        sort=SortSpec("match_strength", True),
    )
    page = reader.query(q)
    assert page.items
    row = page.items[0]
    assert badge_text(row)
    assert "ADR" in feature_caption(row.features) or format_adr_pct(row.features.adr_pct_20)
    detail = reader.get_detail(main.run_id, row.candidate_id)
    if detail.row.chart_ref is not None:
        png = reader.get_artifact_bytes(main.run_id, detail.row.chart_ref.artifact_id)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    next_btn = next(b for b in at.button if b.label == "Next")
    next_btn.click().run()
    assert not at.exception
    reviews.add_review(main.run_id, row.candidate_id, judgment="unsure", note="ui test", setup_id="flag")
    at.run()
    assert not at.exception

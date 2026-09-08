from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_shared_app_configures_pages_once() -> None:
    from screener_loader.ui import app as shared
    from screener_loader.ui import setup_builder
    from screener_loader.vision_ui import results_page

    assert "set_page_config" in inspect.getsource(shared.main)
    assert "set_page_config" not in inspect.getsource(setup_builder.render_builder_page)
    assert "set_page_config" in inspect.getsource(setup_builder.main)
    assert "set_page_config" not in inspect.getsource(results_page)


def test_shared_app_streamlit_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NS_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("NS_VISION_PAGE", "builder")
    app_file = Path(__file__).resolve().parents[1] / "src" / "screener_loader" / "ui" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=30)
    at.run()
    assert not at.exception
    labels = [str(getattr(r, "options", r)) for r in at.radio]
    assert any("Setup builder" in x and "Results" in x for x in labels)
    assert any(b.label == "Create" for b in at.button)

"""Shared Streamlit entry point: setup builder and vision results."""

from __future__ import annotations

from pathlib import Path
import os
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PAGE_BUILDER = "builder"
PAGE_RESULTS = "results"
NAV_KEY = "ns_nav_page"
PAGE_LABELS = ("Setup builder", "Results")


def _repo_root() -> Path:
    return Path(os.environ.get("NS_REPO_ROOT", ".")).resolve()


def _initial_page() -> str:
    raw = os.environ.get("NS_VISION_PAGE", PAGE_BUILDER).strip().lower()
    if raw in {PAGE_RESULTS, "results", "view"}:
        return PAGE_RESULTS
    return PAGE_BUILDER


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Nuanced Screener", layout="wide")
    query_page = str(st.query_params.get("page", "") or "").strip().lower()
    default = PAGE_RESULTS if query_page in {PAGE_RESULTS, "results", "view"} else _initial_page()
    if NAV_KEY not in st.session_state:
        st.session_state[NAV_KEY] = PAGE_LABELS[1] if default == PAGE_RESULTS else PAGE_LABELS[0]

    run_hint = os.environ.get("NS_VISION_RUN_ID") or str(st.query_params.get("run", "") or "")
    if run_hint:
        from screener_loader.vision_ui.components import state_key

        st.session_state.setdefault(state_key("run_id"), run_hint)

    with st.sidebar:
        st.radio("Page", PAGE_LABELS, key=NAV_KEY)
        st.caption("Scans are launched from `ns vision scan`, not from this UI.")

    if st.session_state[NAV_KEY] == PAGE_LABELS[0]:
        from screener_loader.ui.setup_builder import render_builder_page

        render_builder_page()
        return

    from screener_loader.config import LoaderConfig
    from screener_loader.vision.service import default_scan_root
    from screener_loader.vision.store import open_vision_scans
    from screener_loader.vision_ui import render_results_page

    cfg = LoaderConfig(repo_root=_repo_root())
    _store, reader, reviews = open_vision_scans(default_scan_root(cfg))
    render_results_page(reader, reviews)


if __name__ == "__main__":
    main()

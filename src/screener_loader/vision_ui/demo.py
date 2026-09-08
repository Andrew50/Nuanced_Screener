"""Standalone Streamlit demo for the vision results page.

Configure the page here only. The mounted ``render_results_page`` does not call
``st.set_page_config`` and does not launch scans.

    streamlit run src/screener_loader/vision_ui/demo.py

Optional: ``NS_VISION_DEMO_ROOT`` points at a temporary scan root. Data is synthetic.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from screener_loader.vision_ui.results_page import render_results_page  # noqa: E402
from screener_loader.vision_ui.synthetic import seed_synthetic_runs  # noqa: E402

DEMO_ROOT_ENV = "NS_VISION_DEMO_ROOT"


def demo_root() -> Path:
    raw = os.environ.get(DEMO_ROOT_ENV)
    if raw:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = Path(tempfile.gettempdir()) / "nuanced-screener-vision-demo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Vision results (synthetic demo)", layout="wide")
    st.title("Vision scan results")
    st.warning(
        "Synthetic demo data. These are not live market scans, LLM calls, or catalog edits. "
        "Viewing and saving reviews only read/write the local scan root."
    )
    _store, reader, reviews = seed_synthetic_runs(demo_root())
    st.caption(f"Scan root: {demo_root()}")
    render_results_page(reader, reviews)


if __name__ == "__main__":
    main()

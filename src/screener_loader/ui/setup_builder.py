"""Streamlit setup builder. Launch via `ns setups ui`."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import os
import tempfile

import streamlit as st

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.setups.service import SetupService, update_spec_fields
from screener_loader.setups.spec import ChartStyle, SetupCriteria, SetupFilters


def _repo_root() -> Path:
    return Path(os.environ.get("NS_REPO_ROOT", ".")).resolve()


def _service() -> tuple[LoaderConfig, SetupService]:
    cfg = LoaderConfig(repo_root=_repo_root())
    ensure_dirs(cfg.paths)
    return cfg, SetupService(cfg.paths)


def _lines(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in str(text).splitlines() if s.strip())


def _join(items: tuple[str, ...]) -> str:
    return "\n".join(items)


def main() -> None:
    st.set_page_config(page_title="Setup builder", layout="wide")
    cfg, svc = _service()
    specs = svc.list_setups()
    selected = st.session_state.get("setup_id")
    ids = [s.id for s in specs]
    if selected not in ids:
        selected = ids[0] if ids else None
        st.session_state["setup_id"] = selected

    col_list, col_def, col_ex = st.columns([1.0, 1.45, 1.25], gap="large")

    with col_list:
        st.subheader("Setups")
        new_id = st.text_input("New id", placeholder="episodic_pivot")
        new_name = st.text_input("Name", placeholder="Episodic Pivot")
        if st.button("Create", type="primary"):
            try:
                spec = svc.create(new_id.strip(), new_name.strip() or new_id.strip())
                st.session_state["setup_id"] = spec.id
                st.rerun()
            except Exception as e:
                st.error(str(e))

        st.divider()
        for spec in specs:
            label = f"{spec.name}  ({'on' if spec.enabled else 'off'})"
            if st.button(label, key=f"sel_{spec.id}", use_container_width=True):
                st.session_state["setup_id"] = spec.id
                st.rerun()

    if not selected:
        with col_def:
            st.info("Create a setup to begin.")
        return

    spec = svc.get(selected)
    examples = svc.load_examples(spec.id)

    with col_def:
        st.subheader("Setup")
        name = st.text_input("Name", value=spec.name, key=f"name_{spec.id}")
        enabled = st.toggle("Enabled", value=spec.enabled)
        timeframe = st.selectbox("Timeframe", ["1d"], index=0, disabled=True)
        lookback = st.number_input("Lookback bars", min_value=2, max_value=400, value=int(spec.lookback_bars))
        description = st.text_area("Description", value=spec.description, height=90)
        required = st.text_area("Required (one per line)", value=_join(spec.criteria.required), height=80)
        preferred = st.text_area("Preferred (one per line)", value=_join(spec.criteria.preferred), height=80)
        disqualifiers = st.text_area("Disqualifiers (one per line)", value=_join(spec.criteria.disqualifiers), height=80)
        llm_notes = st.text_area("LLM notes", value=spec.llm_notes, height=70)

        st.markdown("**Filters**")
        g = svc.load_global_filters()
        st.caption(
            f"Global floor: min price {g.min_price} · min 20d dollar vol {g.min_dollar_vol_20d}"
        )
        min_price = st.number_input(
            "Min price (setup, optional tighten)",
            value=float(spec.filters.min_price) if spec.filters.min_price is not None else 0.0,
            min_value=0.0,
        )
        use_min_price = st.checkbox("Apply min price", value=spec.filters.min_price is not None)
        min_dv = st.number_input(
            "Min 20d dollar volume",
            value=float(spec.filters.min_dollar_vol_20d) if spec.filters.min_dollar_vol_20d is not None else 0.0,
            min_value=0.0,
            step=100000.0,
            format="%.0f",
        )
        use_min_dv = st.checkbox("Apply dollar volume", value=spec.filters.min_dollar_vol_20d is not None)
        min_adr = st.number_input(
            "Min ADR % (20d)",
            value=float(spec.filters.min_adr_pct_20) * 100.0 if spec.filters.min_adr_pct_20 is not None else 0.0,
            min_value=0.0,
            step=0.1,
            help="Stored as a fraction. 4 means 4% = 0.04. Formula: mean((H-L)/prior close) over 20 bars.",
        )
        use_min_adr = st.checkbox("Apply ADR", value=spec.filters.min_adr_pct_20 is not None)
        st.text_input("Market cap", value="unavailable", disabled=True)

        volume = st.checkbox("Show volume", value=spec.chart.volume)
        mas_text = st.text_input("Moving averages", value=",".join(str(x) for x in spec.chart.moving_averages))

        if st.button("Save", type="primary"):
            try:
                mas = tuple(int(x.strip()) for x in mas_text.split(",") if x.strip())
                updated = update_spec_fields(
                    spec,
                    name=name,
                    enabled=enabled,
                    lookback_bars=int(lookback),
                    description=description,
                    llm_notes=llm_notes,
                    criteria=SetupCriteria(
                        required=_lines(required),
                        preferred=_lines(preferred),
                        disqualifiers=_lines(disqualifiers),
                    ),
                    filters=SetupFilters(
                        min_price=float(min_price) if use_min_price else None,
                        min_dollar_vol_20d=float(min_dv) if use_min_dv else None,
                        min_adr_pct_20=(float(min_adr) / 100.0) if use_min_adr else None,
                    ),
                    chart=ChartStyle(volume=bool(volume), moving_averages=mas or spec.chart.moving_averages),
                    timeframe=timeframe,
                )
                svc.save(updated)
                st.success("Saved")
                st.rerun()
            except Exception as e:
                st.error(str(e))

        with st.expander("Compiled prompt"):
            st.code(svc.compile_prompt(spec.id).text)

    with col_ex:
        st.subheader("Examples")
        mode = st.radio("Add", ["Ticker/date", "Upload image"], horizontal=True)
        polarity = st.radio("Polarity", ["positive", "negative"], horizontal=True)
        quality = st.selectbox("Quality", ["canonical", "decent", "edge_case", "near_miss"])
        note = st.text_input("Notes")

        if mode == "Ticker/date":
            ticker = st.text_input("Ticker", placeholder="NVDA")
            asof = st.date_input("Date", value=date.today())
            if ticker:
                try:
                    from screener_loader.setups.charts import load_ohlcv_window, render_chart_png

                    df = load_ohlcv_window(cfg, ticker, asof, spec.lookback_bars)
                    png = render_chart_png(df, ticker=ticker.upper(), asof_date=asof, style=spec.chart)
                    st.image(png, caption=f"{ticker.upper()} {asof.isoformat()}")
                except Exception as e:
                    st.caption(str(e))
            if st.button("Add example"):
                try:
                    svc.add_market_window_example(
                        spec.id,
                        ticker=ticker,
                        asof_date=asof,
                        polarity=polarity,
                        quality=quality,
                        note=note,
                    )
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        else:
            uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
            if uploaded is not None:
                st.image(uploaded.getvalue())
            if st.button("Add image example"):
                if uploaded is None:
                    st.error("Choose an image")
                else:
                    suffix = Path(uploaded.name).suffix or ".png"
                    tmp_path: Path | None = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(uploaded.getvalue())
                            tmp_path = Path(tmp.name)
                        svc.add_image_example(
                            spec.id,
                            tmp_path,
                            polarity=polarity,
                            quality=quality,
                            note=note,
                        )
                    except Exception as e:
                        st.error(str(e))
                    else:
                        st.rerun()
                    finally:
                        if tmp_path is not None:
                            tmp_path.unlink(missing_ok=True)

        st.divider()
        pos = [e for e in examples if e.polarity == "positive"]
        neg = [e for e in examples if e.polarity == "negative"]
        st.markdown("**Positive**")
        _render_example_list(cfg, spec, pos)
        st.markdown("**Near misses / negatives**")
        _render_example_list(cfg, spec, neg)


def _render_example_list(cfg: LoaderConfig, spec, examples) -> None:
    if not examples:
        st.caption("None")
        return
    for ex in examples:
        if ex.type == "market_window":
            st.write(f"{ex.ticker}  {ex.date}  [{ex.quality or '-'}]")
        else:
            st.write(f"{ex.path}  [{ex.quality or '-'}]")
        if ex.note:
            st.caption(ex.note)
        try:
            from screener_loader.setups.charts import render_example_png

            st.image(render_example_png(cfg, spec, ex), use_container_width=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()

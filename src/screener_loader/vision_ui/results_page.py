"""Mountable Streamlit results page. Does not set page config or launch scans."""

from __future__ import annotations

from typing import Any

from screener_loader.vision.query import FilesystemResultReader, NeighborPosition
from screener_loader.vision.reviews import FilesystemReviewStore
from screener_loader.vision.types import SetupSnapshot, VisionError

from .components import (
    DEFAULT_PAGE_SIZE,
    JUDGMENT_LABELS,
    VIEW_LABELS,
    VIEW_MATCHES,
    badge_text,
    build_result_query,
    feature_caption,
    format_adr_pct,
    format_dollar_volume,
    format_price,
    md_escape_dollars,
    reconcile_selected_candidate,
    run_status_label,
    state_key,
    unavailable_text,
)

_STREAMLIT = None


def _st():
    global _STREAMLIT
    if _STREAMLIT is None:
        import streamlit as st

        _STREAMLIT = st
    return _STREAMLIT


def render_results_page(
    reader: FilesystemResultReader,
    reviews: FilesystemReviewStore,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> None:
    """List/detail viewer over persisted runs. Refresh reads the store only."""

    st = _st()
    runs = reader.list_runs()
    if not runs:
        st.info("No saved vision runs in this scan root.")
        return

    run_labels = {
        run.run_id: (
            f"{run.created_at.strftime('%Y-%m-%d %H:%M')} UTC · {run.status} · {run.config.model}"
            f"{' · synthetic' if run.synthetic else ''} · {run.run_id}"
        )
        for run in runs
    }
    run_ids = [r.run_id for r in runs]
    if state_key("view") not in st.session_state:
        st.session_state[state_key("view")] = VIEW_MATCHES
    if state_key("run_id") not in st.session_state or st.session_state[state_key("run_id")] not in run_ids:
        st.session_state[state_key("run_id")] = run_ids[0]

    top = st.columns([2.4, 1.2, 0.7])
    with top[0]:
        run_id = st.selectbox(
            "Run",
            options=run_ids,
            format_func=lambda rid: run_labels.get(rid, rid),
            key=state_key("run_id"),
        )
    with top[2]:
        if st.button("Refresh", key=state_key("refresh")):
            st.rerun()

    prev_run = st.session_state.get(state_key("prev_run_id"))
    if prev_run is not None and prev_run != run_id:
        st.session_state[state_key("candidate_id")] = None
        st.session_state[state_key("page")] = 1
    st.session_state[state_key("prev_run_id")] = run_id

    try:
        stored = reader.load_run(run_id)
        frozen = reader.load_frozen_inputs(run_id)
    except VisionError as exc:
        st.error(str(exc))
        return

    counts = reader.extended_counts(run_id)
    asof_dates = sorted({c.asof_date for c in frozen.candidates})
    asof_label = ", ".join(d.isoformat() for d in asof_dates) if asof_dates else "n/a"
    st.caption(
        f"{run_status_label(stored.status, synthetic=stored.synthetic, mode=stored.config.mode)} · "
        f"chart session {asof_label} · lookback {frozen.profile.lookback_bars} bars · "
        f"{counts.candidates} candidates · {counts.completed} completed · "
        f"{counts.any_match} any-match · {counts.setup_matches} setup-matches · "
        f"{counts.uncertain} uncertain · {counts.error} error · {counts.skipped} skipped · "
        f"{counts.reviewed_pairs} reviewed pairs"
    )
    diag = frozen.diagnostics
    st.caption(
        "Eligibility diagnostics: "
        f"universe {unavailable_text(diag.universe_count, unavailable=diag.unavailable, name='universe_count')} · "
        f"input {unavailable_text(diag.input_count, unavailable=diag.unavailable, name='input_count')} · "
        f"eligible union {unavailable_text(diag.eligible_union_count, unavailable=diag.unavailable, name='eligible_union_count')} · "
        f"not eligible {unavailable_text(diag.not_eligible_count, unavailable=diag.unavailable, name='not_eligible_count')}"
    )

    view = st.radio(
        "View",
        options=list(VIEW_LABELS.keys()),
        format_func=lambda key: VIEW_LABELS[key],
        horizontal=True,
        key=state_key("view"),
    )
    setups = list(frozen.setups)
    setup_options = [s.setup_id for s in setups]
    filters = st.columns([1.2, 1.4, 0.9, 0.9, 1.0, 0.9])
    with filters[0]:
        ticker_query = st.text_input("Ticker search", key=state_key("ticker"))
    with filters[1]:
        selected_setups = st.multiselect(
            "Setups (ANY)",
            options=setup_options,
            format_func=lambda sid: next((s.name for s in setups if s.setup_id == sid), sid),
            key=state_key("setup_ids"),
        )
    with filters[2]:
        strength = st.selectbox(
            "Min strength",
            options=("any", 1, 2, 3),
            key=state_key("min_strength"),
        )
    with filters[3]:
        review_state = st.selectbox(
            "Review",
            options=("any", "unreviewed", "agree", "disagree", "unsure"),
            key=state_key("review_state"),
        )
    with filters[4]:
        sort_key = st.selectbox(
            "Sort",
            options=("match_strength", "ticker", "asof_date", "candidate_id"),
            key=state_key("sort_key"),
        )
    with filters[5]:
        descending = st.checkbox("Descending", value=True, key=state_key("sort_desc"))

    min_strength = None if strength == "any" else int(strength)
    page = int(st.session_state.get(state_key("page")) or 1)
    query, statuses = build_result_query(
        run_id=run_id,
        view=view,
        setup_ids=selected_setups,
        min_match_strength=min_strength,
        review_state=review_state,
        sort_key=sort_key,
        descending=descending,
        page=page,
        page_size=page_size,
    )
    ordered_ids = reader.ordered_ids(query, ticker_query=ticker_query, statuses=statuses)
    selected_id = reconcile_selected_candidate(
        st.session_state.get(state_key("candidate_id")),
        ordered_ids,
    )
    st.session_state[state_key("candidate_id")] = selected_id
    if selected_id:
        pos = reader.neighbor(query, selected_id, ticker_query=ticker_query, statuses=statuses)
        if pos is not None and view == st.session_state.get(state_key("view")):
            # Keep the selected row on-screen after filter changes; Prev/Next also updates page.
            if st.session_state.get(state_key("snap_page")):
                page = pos.page
                st.session_state[state_key("page")] = page
                st.session_state[state_key("snap_page")] = False
                query, statuses = build_result_query(
                    run_id=run_id,
                    view=view,
                    setup_ids=selected_setups,
                    min_match_strength=min_strength,
                    review_state=review_state,
                    sort_key=sort_key,
                    descending=descending,
                    page=page,
                    page_size=page_size,
                )
    else:
        page = 1
        st.session_state[state_key("page")] = 1

    pages = reader.list_pages(query, ticker_query=ticker_query, statuses=statuses)
    result_page = reader.query(query, ticker_query=ticker_query, statuses=statuses)

    list_col, detail_col = st.columns([1.05, 1.55], gap="large")
    with list_col:
        st.subheader("Candidates")
        if not result_page.items and not ordered_ids:
            st.info("No candidates match these filters.")
        elif not result_page.items:
            st.info("This page is empty. Use Previous/Next or another page.")
        for row in result_page.items:
            selected = row.candidate_id == selected_id
            label = f"{'▸ ' if selected else ''}{row.ticker}  {badge_text(row)}"
            if st.button(label, key=state_key(f"pick_{row.candidate_id}"), use_container_width=True):
                st.session_state[state_key("candidate_id")] = row.candidate_id
                st.session_state[state_key("snap_page")] = False
                st.rerun()
            st.caption(
                md_escape_dollars(
                    f"{row.asof_date.isoformat()} · {format_price(row.features.close)} · "
                    f"{format_dollar_volume(row.features.dollar_vol_avg_20)} · ADR {format_adr_pct(row.features.adr_pct_20)}"
                    f" · review {row.review_state}"
                )
            )
        pager = st.columns([1, 1, 1.4])
        with pager[0]:
            if st.button("Prev page", disabled=not result_page.has_prev, key=state_key("prev_page")):
                st.session_state[state_key("page")] = result_page.prev_page or 1
                st.rerun()
        with pager[1]:
            if st.button("Next page", disabled=not result_page.has_next, key=state_key("next_page")):
                st.session_state[state_key("page")] = result_page.next_page or page
                st.rerun()
        with pager[2]:
            st.caption(
                f"Page {result_page.page} of {max(len(pages), 1)} · {result_page.total_candidates} in view"
            )
        export_q, export_statuses = query, statuses
        st.download_button(
            "Export matching candidates (CSV)",
            data=reader.export_csv(export_q, ticker_query=ticker_query, statuses=export_statuses, table="candidates"),
            file_name=f"{run_id}-candidates.csv",
            mime="text/csv",
            key=state_key("export_cand"),
        )
        st.download_button(
            "Export matching assessments (CSV)",
            data=reader.export_csv(export_q, ticker_query=ticker_query, statuses=export_statuses, table="assessments"),
            file_name=f"{run_id}-assessments.csv",
            mime="text/csv",
            key=state_key("export_assess"),
        )
        st.caption(reader.export_query(export_q, ticker_query=ticker_query, statuses=export_statuses).flattening)

    with detail_col:
        st.subheader("Detail")
        if not selected_id:
            st.info("Select a candidate from the list.")
            return
        try:
            detail = reader.get_detail(run_id, selected_id)
        except VisionError as exc:
            st.error(str(exc))
            return
        pos = reader.neighbor(query, selected_id, ticker_query=ticker_query, statuses=statuses)
        _render_nav(st, pos)
        row = detail.row
        st.markdown(f"**{row.ticker}** · {row.asof_date.isoformat()} · {row.status}")
        st.caption(md_escape_dollars(feature_caption(row.features)))
        if row.matched_setups:
            st.markdown("**Matched setups:** " + ", ".join(f"{b.setup_name} ({b.match_strength})" for b in row.matched_setups))
        _render_chart(st, reader, run_id, row)
        snapshots = {s.setup_id: s for s in frozen.setups}
        current_reviews = {r.setup_id: r for r in reviews.current_reviews_for_candidate(run_id, selected_id)}
        if detail.assessments:
            for assessment in detail.assessments:
                snap = snapshots.get(assessment.setup_id)
                _render_assessment(st, assessment, snap, current_reviews.get(assessment.setup_id), reviews, run_id, selected_id)
        elif row.error:
            st.error(f"Execution {row.error.kind}: {row.error.message}")
            st.caption("Refusals, timeouts, invalid output, and unavailable inputs are not no-match verdicts.")
        else:
            st.info("No model assessments stored (pending, dry-run, or skipped).")
        with st.expander("Frozen eligibility and attempts"):
            st.write(
                {
                    "eligible_setup_ids": list(detail.eligible_setup_ids),
                    "review_history": [
                        {
                            "review_id": r.review_id,
                            "setup_id": r.setup_id,
                            "judgment": r.judgment,
                            "note": r.note,
                            "supersedes": r.supersedes_review_id,
                        }
                        for r in detail.reviews
                    ],
                    "attempts": [
                        {
                            "attempt_id": a.attempt_id,
                            "accepted": a.accepted,
                            "error": a.error.kind if a.error else None,
                            "usage": None
                            if a.usage is None
                            else {
                                "input": a.usage.input_tokens,
                                "output": a.usage.output_tokens,
                            },
                        }
                        for a in detail.attempts
                    ],
                }
            )


def _render_nav(st: Any, pos: NeighborPosition | None) -> None:
    cols = st.columns([1, 1, 1.6])
    prev_disabled = pos is None or pos.prev_id is None
    next_disabled = pos is None or pos.next_id is None
    with cols[0]:
        if st.button("Previous", disabled=prev_disabled, key=state_key("prev_item")):
            st.session_state[state_key("candidate_id")] = pos.prev_id if pos else None
            st.session_state[state_key("snap_page")] = True
            st.rerun()
    with cols[1]:
        if st.button("Next", disabled=next_disabled, key=state_key("next_item")):
            st.session_state[state_key("candidate_id")] = pos.next_id if pos else None
            st.session_state[state_key("snap_page")] = True
            st.rerun()
    with cols[2]:
        if pos is None:
            st.caption("Not in the current filtered sequence")
        else:
            st.caption(f"{pos.position} of {pos.total}")


def _render_chart(st: Any, reader: FilesystemResultReader, run_id: str, row) -> None:
    if row.chart_ref is None:
        st.warning("No saved chart for this candidate. Missing charts are not regenerated on view.")
        return
    try:
        png = reader.get_artifact_bytes(run_id, row.chart_ref.artifact_id)
    except VisionError as exc:
        st.error(f"Saved chart unavailable: {exc}")
        return
    st.image(png, caption=f"Persisted chart {row.chart_ref.width}×{row.chart_ref.height}", use_container_width=True)


def _render_assessment(
    st: Any,
    assessment,
    snap: SetupSnapshot | None,
    current_review,
    reviews: FilesystemReviewStore,
    run_id: str,
    candidate_id: str,
) -> None:
    name = snap.name if snap is not None else assessment.setup_id
    strength = f" · strength {assessment.match_strength}" if assessment.match_strength is not None else ""
    st.markdown(f"**{name}** · model `{assessment.verdict}`{strength}")
    st.write(assessment.reason)
    if assessment.violated_required_rule_ids:
        labels = []
        for rid in assessment.violated_required_rule_ids:
            text = rid
            if snap is not None:
                for rule in snap.rules:
                    if rule.rule_id == rid:
                        text = f"{rid}: {rule.text}"
                        break
            labels.append(text)
        st.caption("Violated required rules: " + "; ".join(labels))
    if assessment.missing_evidence:
        st.caption("Missing evidence: " + "; ".join(assessment.missing_evidence))
    if snap is not None:
        with st.expander(f"Frozen criteria · {name}"):
            st.caption("These texts come from the run snapshot, not the current catalog.")
            st.markdown("**Required**")
            st.write("\n".join(f"- {r.text}" for r in snap.rules_of("required")) or "_none_")
            st.markdown("**Preferred**")
            st.write("\n".join(f"- {r.text}" for r in snap.rules_of("preferred")) or "_none_")
            st.markdown("**Disqualifiers**")
            st.write("\n".join(f"- {r.text}" for r in snap.rules_of("disqualifier")) or "_none_")
    st.caption("Human review (separate from the model output above)")
    judgment_key = state_key(f"judgment_{candidate_id}_{assessment.setup_id}")
    note_key = state_key(f"note_{candidate_id}_{assessment.setup_id}")
    default_judgment = current_review.judgment if current_review is not None else "unsure"
    options = ("agree", "disagree", "unsure")
    st.radio(
        f"Review {name}",
        options=options,
        format_func=lambda j: JUDGMENT_LABELS[j],
        index=options.index(default_judgment) if default_judgment in options else 2,
        key=judgment_key,
        horizontal=True,
    )
    st.text_area("Notes", value=current_review.note if current_review else "", key=note_key, height=70)
    if current_review is not None:
        st.caption(f"Current review {current_review.judgment} at {current_review.created_at.isoformat()}")
    if st.button(f"Save review · {name}", key=state_key(f"save_{candidate_id}_{assessment.setup_id}")):
        reviews.add_review(
            run_id,
            candidate_id,
            judgment=st.session_state[judgment_key],
            note=st.session_state.get(note_key) or "",
            setup_id=assessment.setup_id,
        )
        st.rerun()

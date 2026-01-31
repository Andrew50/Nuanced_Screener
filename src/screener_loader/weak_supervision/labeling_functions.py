from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


Vote = int  # -1, 0, +1


@dataclass(frozen=True)
class LabelingFunction:
    name: str
    apply_one: Callable[[dict], Vote]

    def __call__(self, row: dict) -> Vote:
        v = int(self.apply_one(row))
        if v not in (-1, 0, 1):
            raise ValueError(f"LF {self.name} returned invalid vote {v} (expected -1/0/+1)")
        return v


def _finite(*xs: float) -> bool:
    return all(np.isfinite(float(x)) for x in xs)


def _get(d: dict, k: str) -> float:
    try:
        return float(d.get(k))
    except Exception:
        return float("nan")


def make_flag_lfs() -> list[LabelingFunction]:
    """
    Heuristics for a present-tense flag state at asof_date.

    Uses future bars only as validation (e.g. confirming a clean breakout happens later),
    not as the definition of the state itself.
    """

    def pos_impulse_and_tight(d: dict) -> Vote:
        imp = _get(d, "impulse_return")
        cr = _get(d, "cons_range_pct")
        vs = _get(d, "cons_vol_slope")
        if not _finite(imp, cr, vs):
            return 0
        if imp > 0.12 and cr < 0.08 and vs < 0:
            return 1
        return 0

    def pos_breakout_validates(d: dict) -> Vote:
        cons_high = _get(d, "cons_high")
        asof_open = _get(d, "asof_open")
        fut_max = _get(d, "future_max_close")
        if not _finite(cons_high, asof_open, fut_max) or cons_high <= 0:
            return 0
        # Present tense: not already broken out at open, but future validates a clean move.
        if asof_open <= cons_high * 1.01 and fut_max >= cons_high * 1.03:
            return 1
        return 0

    def pos_shallow_retrace(d: dict) -> Vote:
        ch = _get(d, "cons_high")
        cl = _get(d, "cons_low")
        imp = _get(d, "impulse_return")
        if not _finite(ch, cl, imp) or ch <= 0:
            return 0
        retr = cl / ch
        if imp > 0.10 and retr >= 0.85:
            return 1
        return 0

    def pos_vol_shock_on_tight_base(d: dict) -> Vote:
        cr = _get(d, "cons_range_pct")
        shock = _get(d, "volume_shock")
        gap = _get(d, "gap_pct")
        if not _finite(cr, shock, gap):
            return 0
        if cr < 0.06 and shock > 1.8 and gap > -0.02:
            return 1
        return 0

    def neg_too_wide(d: dict) -> Vote:
        cr = _get(d, "cons_range_pct")
        if not _finite(cr):
            return 0
        if cr > 0.15:
            return -1
        return 0

    def neg_deep_retrace(d: dict) -> Vote:
        ch = _get(d, "cons_high")
        cl = _get(d, "cons_low")
        if not _finite(ch, cl) or ch <= 0:
            return 0
        retr = cl / ch
        if retr < 0.75:
            return -1
        return 0

    def neg_bad_gap(d: dict) -> Vote:
        gap = _get(d, "gap_pct")
        if not _finite(gap):
            return 0
        if gap < -0.03:
            return -1
        return 0

    def neg_breaks_down_soon(d: dict) -> Vote:
        cl = _get(d, "cons_low")
        fut_min = _get(d, "future_min_close")
        if not _finite(cl, fut_min) or cl <= 0:
            return 0
        if fut_min <= cl * 0.97:
            return -1
        return 0

    return [
        LabelingFunction("flag_pos_impulse_and_tight", pos_impulse_and_tight),
        LabelingFunction("flag_pos_breakout_validates", pos_breakout_validates),
        LabelingFunction("flag_pos_shallow_retrace", pos_shallow_retrace),
        LabelingFunction("flag_pos_vol_shock_on_tight_base", pos_vol_shock_on_tight_base),
        LabelingFunction("flag_neg_too_wide", neg_too_wide),
        LabelingFunction("flag_neg_deep_retrace", neg_deep_retrace),
        LabelingFunction("flag_neg_bad_gap", neg_bad_gap),
        LabelingFunction("flag_neg_breaks_down_soon", neg_breaks_down_soon),
    ]


def make_gap_lfs(subpattern: str) -> list[LabelingFunction]:
    """
    Daily-only gap labelers.

    subpattern:
    - gap_go: large positive gap that holds / continues (teacher uses same-day close/low as validation)
    - gap_fade: large positive gap that fades / fills (teacher uses same-day close/low as validation)
    """
    sp = (subpattern or "").strip().lower()
    if sp not in {"gap_go", "gap_fade"}:
        raise ValueError("subpattern must be gap_go or gap_fade")

    def _is_gap(d: dict, thr: float) -> bool:
        gap = _get(d, "gap_pct")
        return bool(np.isfinite(gap) and gap >= float(thr))

    def _range_pos(d: dict) -> float:
        hi = _get(d, "asof_high")
        lo = _get(d, "asof_low")
        cl = _get(d, "asof_close")
        if not _finite(hi, lo, cl) or hi <= lo:
            return float("nan")
        return float((cl - lo) / (hi - lo))

    if sp == "gap_go":

        def pos_gap_hold(d: dict) -> Vote:
            if not _is_gap(d, 0.03):
                return 0
            prev = _get(d, "prev_close")
            lo = _get(d, "asof_low")
            op = _get(d, "asof_open")
            cl = _get(d, "asof_close")
            if not _finite(prev, lo, op, cl):
                return 0
            if lo >= prev and cl >= op:
                return 1
            return 0

        def pos_gap_strong_close(d: dict) -> Vote:
            if not _is_gap(d, 0.04):
                return 0
            rp = _range_pos(d)
            if np.isfinite(rp) and rp >= 0.7:
                return 1
            return 0

        def pos_gap_volume_shock(d: dict) -> Vote:
            if not _is_gap(d, 0.03):
                return 0
            shock = _get(d, "volume_shock")
            if np.isfinite(shock) and shock >= 1.5:
                return 1
            return 0

        def neg_gap_fills(d: dict) -> Vote:
            if not _is_gap(d, 0.03):
                return 0
            prev = _get(d, "prev_close")
            cl = _get(d, "asof_close")
            if not _finite(prev, cl):
                return 0
            if cl <= prev:
                return -1
            return 0

        def neg_red_day(d: dict) -> Vote:
            if not _is_gap(d, 0.03):
                return 0
            op = _get(d, "asof_open")
            cl = _get(d, "asof_close")
            if not _finite(op, cl):
                return 0
            if cl < op:
                return -1
            return 0

        return [
            LabelingFunction("gap_go_pos_gap_hold", pos_gap_hold),
            LabelingFunction("gap_go_pos_gap_strong_close", pos_gap_strong_close),
            LabelingFunction("gap_go_pos_gap_volume_shock", pos_gap_volume_shock),
            LabelingFunction("gap_go_neg_gap_fills", neg_gap_fills),
            LabelingFunction("gap_go_neg_red_day", neg_red_day),
        ]

    # gap_fade

    def pos_gap_fills(d: dict) -> Vote:
        if not _is_gap(d, 0.03):
            return 0
        prev = _get(d, "prev_close")
        lo = _get(d, "asof_low")
        if not _finite(prev, lo):
            return 0
        if lo <= prev:
            return 1
        return 0

    def pos_close_below_prev(d: dict) -> Vote:
        if not _is_gap(d, 0.03):
            return 0
        prev = _get(d, "prev_close")
        cl = _get(d, "asof_close")
        if not _finite(prev, cl):
            return 0
        if cl <= prev:
            return 1
        return 0

    def pos_red_day(d: dict) -> Vote:
        if not _is_gap(d, 0.03):
            return 0
        op = _get(d, "asof_open")
        cl = _get(d, "asof_close")
        if not _finite(op, cl):
            return 0
        if cl < op:
            return 1
        return 0

    def neg_strong_gap_hold(d: dict) -> Vote:
        if not _is_gap(d, 0.03):
            return 0
        prev = _get(d, "prev_close")
        lo = _get(d, "asof_low")
        op = _get(d, "asof_open")
        cl = _get(d, "asof_close")
        if not _finite(prev, lo, op, cl):
            return 0
        if lo >= prev and cl >= op:
            return -1
        return 0

    return [
        LabelingFunction("gap_fade_pos_gap_fills", pos_gap_fills),
        LabelingFunction("gap_fade_pos_close_below_prev", pos_close_below_prev),
        LabelingFunction("gap_fade_pos_red_day", pos_red_day),
        LabelingFunction("gap_fade_neg_strong_gap_hold", neg_strong_gap_hold),
    ]


def apply_labeling_functions(
    sample_features: pd.DataFrame,
    *,
    lfs: list[LabelingFunction],
) -> pd.DataFrame:
    """
    Apply LFs to a per-sample features table.

    Returns a wide matrix:
    - sample_id
    - one int column per LF name with values in {-1,0,+1}
    """
    if sample_features.empty:
        return pd.DataFrame({"sample_id": []})
    if "sample_id" not in sample_features.columns:
        raise ValueError("sample_features must include sample_id")
    if not lfs:
        return sample_features[["sample_id"]].copy()

    rows = []
    for r in sample_features.to_dict(orient="records"):
        out = {"sample_id": str(r["sample_id"])}
        for lf in lfs:
            out[lf.name] = int(lf(r))
        rows.append(out)
    return pd.DataFrame(rows)


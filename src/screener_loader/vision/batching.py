"""Deterministic packing, split, and retry delay helpers for the runner."""

from __future__ import annotations

from typing import Sequence, TypeVar

from .types import CandidateInput, ScanConfig

T = TypeVar("T")


def sort_candidates(candidates: Sequence[CandidateInput]) -> list[CandidateInput]:
    return sorted(candidates, key=lambda c: c.candidate_id)


def chunk_candidates(candidates: Sequence[CandidateInput], batch_size: int) -> list[tuple[CandidateInput, ...]]:
    size = max(1, int(batch_size))
    ordered = sort_candidates(candidates)
    return [tuple(ordered[i : i + size]) for i in range(0, len(ordered), size)]


def split_candidates(candidates: Sequence[CandidateInput]) -> tuple[tuple[CandidateInput, ...], tuple[CandidateInput, ...]]:
    ordered = sort_candidates(candidates)
    if len(ordered) <= 1:
        return tuple(ordered), ()
    mid = len(ordered) // 2
    return tuple(ordered[:mid]), tuple(ordered[mid:])


def retry_delay_seconds(
    attempt_index: int,
    config: ScanConfig,
    *,
    retry_after: float | None = None,
    rng_random: float = 0.0,
) -> float:
    if retry_after is not None:
        base = max(0.0, float(retry_after))
    else:
        base = float(config.retry_backoff_seconds) * (2 ** max(0, int(attempt_index)))
        base = min(base, float(config.retry_backoff_max_seconds))
    jitter = max(0.0, float(config.jitter_ratio)) * base * max(0.0, min(1.0, float(rng_random)))
    return base + jitter

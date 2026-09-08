"""Append-only human reviews, separate from model predictions and the catalog."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .store import (
    Clock,
    FilesystemRunStore,
    atomic_write_json,
    format_dt,
    is_temp_name,
    iter_json_files,
    parse_dt,
    read_json,
    safe_segment,
    utcnow,
)
from .types import ReviewRecord, VisionError

_MISSING = object()

VALID_JUDGMENTS = frozenset({"agree", "disagree", "unsure"})


def encode_review(record: ReviewRecord) -> dict[str, Any]:
    return {
        "review_id": record.review_id,
        "run_id": record.run_id,
        "candidate_id": record.candidate_id,
        "setup_id": record.setup_id,
        "judgment": record.judgment,
        "note": record.note,
        "created_at": format_dt(record.created_at),
        "supersedes_review_id": record.supersedes_review_id,
    }


def decode_review(payload: dict[str, Any]) -> ReviewRecord:
    return ReviewRecord(
        review_id=str(payload["review_id"]),
        run_id=str(payload["run_id"]),
        candidate_id=str(payload["candidate_id"]),
        setup_id=payload.get("setup_id"),
        judgment=payload["judgment"],
        note=str(payload.get("note") or ""),
        created_at=parse_dt(payload["created_at"]),
        supersedes_review_id=payload.get("supersedes_review_id"),
    )


class FilesystemReviewStore:
    """Append-only reviews keyed by run/candidate/setup. Never mutates prior rows."""

    def __init__(self, scan_root: Path, *, clock: Clock | None = None, store: FilesystemRunStore | None = None) -> None:
        self.scan_root = Path(scan_root)
        self.scan_root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or utcnow
        self._store = store or FilesystemRunStore(self.scan_root, clock=clock)
        self._lock = Lock()

    def add_review(
        self,
        run_id: str,
        candidate_id: str,
        *,
        judgment: str,
        note: str = "",
        setup_id: str | None = None,
        expected_current_id: object = _MISSING,
    ) -> ReviewRecord:
        if judgment not in VALID_JUDGMENTS:
            raise VisionError(f"judgment must be one of {sorted(VALID_JUDGMENTS)}")
        directory = self._store.run_dir(run_id)
        if not directory.exists():
            raise VisionError(f"unknown run {run_id}")
        with self._lock:
            current = self.current_review(run_id, candidate_id, setup_id=setup_id)
            if expected_current_id is not _MISSING:
                current_id = current.review_id if current else None
                if current_id != expected_current_id:
                    raise VisionError(
                        "conflicting review revision: "
                        f"expected {expected_current_id!r}, current is {current_id!r}"
                    )
            record = ReviewRecord(
                review_id=f"rev-{uuid4().hex}",
                run_id=run_id,
                candidate_id=candidate_id,
                setup_id=setup_id,
                judgment=judgment,  # type: ignore[arg-type]
                note=str(note or ""),
                created_at=self._clock(),
                supersedes_review_id=current.review_id if current else None,
            )
            path = directory / "reviews" / f"{safe_segment(record.review_id)}.json"
            atomic_write_json(path, encode_review(record))
            return record

    def current_review(
        self,
        run_id: str,
        candidate_id: str,
        *,
        setup_id: str | None = None,
    ) -> ReviewRecord | None:
        recs = [
            r
            for r in self._list_candidate_reviews(run_id, candidate_id)
            if r.setup_id == setup_id
        ]
        return _latest_review(recs)

    def list_reviews(self, run_id: str, candidate_id: str) -> tuple[ReviewRecord, ...]:
        recs = list(self._list_candidate_reviews(run_id, candidate_id))
        recs.sort(key=lambda r: (r.created_at, r.review_id))
        return tuple(recs)

    def list_run_reviews(self, run_id: str) -> tuple[ReviewRecord, ...]:
        directory = self._store.run_dir(run_id)
        if not directory.exists():
            raise VisionError(f"unknown run {run_id}")
        recs = [decode_review(read_json(path)) for path in iter_json_files(directory / "reviews")]
        recs.sort(key=lambda r: (r.created_at, r.review_id))
        return tuple(recs)

    def current_reviews_for_candidate(self, run_id: str, candidate_id: str) -> tuple[ReviewRecord, ...]:
        grouped: dict[str | None, list[ReviewRecord]] = {}
        for rec in self._list_candidate_reviews(run_id, candidate_id):
            grouped.setdefault(rec.setup_id, []).append(rec)
        out = []
        for recs in grouped.values():
            latest = _latest_review(recs)
            if latest is not None:
                out.append(latest)
        out.sort(key=lambda r: (r.setup_id or "", r.created_at, r.review_id))
        return tuple(out)

    def _list_candidate_reviews(self, run_id: str, candidate_id: str) -> tuple[ReviewRecord, ...]:
        directory = self._store.run_dir(run_id)
        if not directory.exists():
            raise VisionError(f"unknown run {run_id}")
        reviews_dir = directory / "reviews"
        if not reviews_dir.is_dir():
            return ()
        recs = []
        for path in iter_json_files(reviews_dir):
            if is_temp_name(path.name):
                continue
            rec = decode_review(read_json(path))
            if rec.candidate_id == candidate_id:
                recs.append(rec)
        return tuple(recs)


def _latest_review(recs: list[ReviewRecord]) -> ReviewRecord | None:
    if not recs:
        return None
    superseded = {r.supersedes_review_id for r in recs if r.supersedes_review_id}
    heads = [r for r in recs if r.review_id not in superseded]
    if not heads:
        heads = list(recs)
    heads.sort(key=lambda r: (r.created_at, r.review_id))
    return heads[-1]

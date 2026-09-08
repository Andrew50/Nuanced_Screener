"""Explicit labeled demo classifier. Never used as a missing-key fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Sequence
from uuid import uuid4

from ..types import (
    Assessment,
    CandidateClassification,
    ClassificationAttempt,
    CompiledRequest,
    PreparedScan,
    TokenUsage,
)

DEMO_MODEL_ID = "demo-vision-classifier"


def _demo_no_match(setup_id: str) -> Assessment:
    return Assessment(
        setup_id=setup_id,
        verdict="no_match",
        match_strength=None,
        reason="Demo classifier: no coiled structure in synthetic labels.",
        violated_required_rule_ids=(),
        missing_evidence=(),
    )


class DemoClassifier:
    """Deterministic local classifier used only when ScanConfig.mode='demo'."""

    provider = "fake"

    @classmethod
    def covering(cls, prepared: PreparedScan) -> "DemoClassifier":
        script = {
            cand.candidate_id: tuple(_demo_no_match(sid) for sid in cand.eligible_setup_ids)
            for cand in prepared.candidates
        }
        return cls(script=script)

    def __init__(self, script: dict[str, Sequence[Assessment]] | None = None) -> None:
        self.script = dict(script or {})
        self.calls = 0
        self.requests: list[CompiledRequest] = []
        self._lock = Lock()

    def classify(self, request: CompiledRequest) -> ClassificationAttempt:
        with self._lock:
            self.calls += 1
            self.requests.append(request)
        now = datetime.now(timezone.utc)
        results: list[CandidateClassification] = []
        for cid in request.candidate_ids:
            assessments = tuple(self.script.get(cid, ()))
            if not assessments:
                assessments = tuple(_demo_no_match(sid) for sid in request.setup_ids)
            results.append(CandidateClassification(candidate_id=cid, assessments=assessments))
        payload = {
            "results": [
                {
                    "candidate_id": row.candidate_id,
                    "assessments": [
                        {
                            "setup_id": a.setup_id,
                            "verdict": a.verdict,
                            "match_strength": a.match_strength,
                            "reason": a.reason,
                            "violated_required_rule_ids": list(a.violated_required_rule_ids),
                            "missing_evidence": list(a.missing_evidence),
                        }
                        for a in row.assessments
                    ],
                }
                for row in results
            ]
        }
        return ClassificationAttempt(
            attempt_id=str(uuid4()),
            batch_id=request.batch_id,
            batch_fingerprint=request.fingerprint,
            candidate_ids=request.candidate_ids,
            setup_ids=request.setup_ids,
            started_at=now,
            ended_at=now,
            provider=self.provider,
            model=request.model or DEMO_MODEL_ID,
            response_id=f"demo-{request.batch_id}",
            usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            retry_after_seconds=None,
            latency_ms=0,
            error=None,
            sanitized_output=payload,
            results=tuple(results),
            accepted=True,
        )

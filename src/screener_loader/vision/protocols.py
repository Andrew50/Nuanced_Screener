"""Public protocols for the vision scan pipeline.

Core contracts do not import OpenAI, Streamlit, or matplotlib.
Implementations live in Agent 1 (render/compile/classify/run), Agent 2 (store/query/review),
and Agent 3 (ScanSource adapters).
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .types import (
    ArtifactRef,
    BarWindow,
    CandidateDetail,
    CandidateInput,
    CandidateResult,
    ChartArtifact,
    ChartProfile,
    ClassificationAttempt,
    CompiledRequest,
    ExampleInput,
    PreparedScan,
    ResultCounts,
    ResultPage,
    ResultQuery,
    ReviewRecord,
    RunSummary,
    ScanConfig,
    ScanOutcome,
    SetupSnapshot,
    StoredRun,
)


class CancelFlag(Protocol):
    def is_set(self) -> bool: ...


class ScanSource(Protocol):
    """Agent 3: bind SetupService, eligibility, last-N bars, and the ticker universe."""

    def prepare(self, config: ScanConfig) -> PreparedScan:
        """
        Build immutable scan inputs.

        Must intersect eligibility with load_universe(config)/tickers.csv, use bulk last-N
        slices, fail closed on market-cap filters, and raise InsufficientLastNError or
        StaleSnapshotError for global problems. Isolated short/stale windows belong in
        PreparedScan.skips, not as invented no-match predictions.
        """
        ...


class ChartRenderer(Protocol):
    """Agent 1: translate bars/profile and delegate to setups.charts. No I/O."""

    def render(
        self,
        window: BarWindow,
        profile: ChartProfile,
        *,
        title: str | None = None,
    ) -> ChartArtifact: ...

    def wrap_upload(self, example: ExampleInput) -> ChartArtifact:
        """Preserve original uploaded image bytes. Do not re-plot."""
        ...


class RequestCompiler(Protocol):
    def compile(
        self,
        *,
        setups: Sequence[SetupSnapshot],
        example_artifacts: Sequence[ChartArtifact],
        examples: Sequence[ExampleInput],
        candidate_artifacts: Sequence[ChartArtifact],
        candidates: Sequence[CandidateInput],
        config: ScanConfig,
        batch_id: str,
    ) -> CompiledRequest:
        """
        Provider-independent multimodal request. Raises OversizeRequestError instead of
        dropping examples, rules, or candidates.
        """
        ...


class Classifier(Protocol):
    """One provider attempt. No retries. The runner owns pacing and recovery."""

    def classify(self, request: CompiledRequest) -> ClassificationAttempt: ...


class RunStore(Protocol):
    """Agent 2: durable run persistence. Runner calls are listed in docs/VISION_CONTRACT.md."""

    def create_run(self, prepared: PreparedScan) -> StoredRun: ...

    def lock_run(self, run_id: str):
        """Return a context manager. Raises RunLockedError if held elsewhere."""
        ...

    def load_run(self, run_id: str) -> StoredRun: ...

    def load_frozen_inputs(self, run_id: str) -> PreparedScan: ...

    def save_artifact(self, run_id: str, artifact: ChartArtifact) -> ArtifactRef: ...

    def get_artifact_bytes(self, run_id: str, artifact_id: str) -> bytes: ...

    def journal_attempt(self, run_id: str, attempt: ClassificationAttempt) -> None: ...

    def recover_attempt(self, run_id: str, batch_fingerprint: str) -> ClassificationAttempt | None:
        """Return the latest accepted attempt for this fingerprint, if stored."""
        ...

    def commit_batch(
        self,
        run_id: str,
        *,
        batch_id: str,
        results: Sequence[CandidateResult],
        attempt: ClassificationAttempt,
    ) -> None:
        """Atomically persist validated candidate results. Idempotent per candidate_id."""
        ...

    def mark_candidates(
        self,
        run_id: str,
        results: Sequence[CandidateResult],
    ) -> None:
        """Record skips/errors without assessments. Does not invent no-match verdicts."""
        ...

    def list_committed_candidate_ids(self, run_id: str) -> frozenset[str]: ...

    def list_candidate_results(self, run_id: str) -> tuple[CandidateResult, ...]: ...

    def update_status(self, run_id: str, status: str) -> StoredRun: ...

    def finalize(self, run_id: str, summary: RunSummary) -> StoredRun: ...

    def export_manifest(self, run_id: str) -> dict: ...


class ResultReader(Protocol):
    """Agent 2: one row per candidate; filters described in VISION_CONTRACT.md."""

    def query(self, q: ResultQuery) -> ResultPage: ...

    def get_detail(self, run_id: str, candidate_id: str) -> CandidateDetail: ...

    def counts(self, run_id: str) -> ResultCounts: ...

    def get_artifact_bytes(self, run_id: str, artifact_id: str) -> bytes: ...

    def get_artifact_ref(self, run_id: str, artifact_id: str) -> ArtifactRef: ...

    def list_pages(self, q: ResultQuery) -> tuple[int, ...]:
        """Page numbers that exist for this query (for prev/next navigation)."""
        ...


class ReviewStore(Protocol):
    def add_review(
        self,
        run_id: str,
        candidate_id: str,
        *,
        judgment: str,
        note: str = "",
        setup_id: str | None = None,
    ) -> ReviewRecord:
        """Append-only. Supersedes the current review for the same (candidate, setup_id)."""
        ...

    def current_review(
        self,
        run_id: str,
        candidate_id: str,
        *,
        setup_id: str | None = None,
    ) -> ReviewRecord | None: ...

    def list_reviews(self, run_id: str, candidate_id: str) -> tuple[ReviewRecord, ...]: ...


class VisionScanner(Protocol):
    def run(
        self,
        prepared: PreparedScan,
        *,
        cancel: CancelFlag | None = None,
    ) -> ScanOutcome: ...

    def resume(
        self,
        run_id: str,
        prepared: PreparedScan | None = None,
        *,
        cancel: CancelFlag | None = None,
    ) -> ScanOutcome: ...

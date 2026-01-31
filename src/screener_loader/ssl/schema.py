from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SSLSchema:
    """
    Strict schema saved alongside pretrained encoder weights.

    This is used to prevent silent mismatch bugs between pretrain and finetune runs.
    """

    schema_version: int = 1
    model_type: str = "ssl_tcn_masked_pretrain"

    # Feature pipeline
    feature_names: list[str] = field(default_factory=list)
    normalization: str = "per_window_zscore"

    # Window policy
    window_max: int = 96
    crop_lengths: list[int] = field(default_factory=lambda: [64, 96])

    # Masking policy
    mask_mode: list[str] = field(default_factory=lambda: ["both"])
    mask_rate_time: float = 0.10
    mask_rate_feat: float = 0.15
    use_mask_token: bool = True
    loss: str = "huber"
    huber_delta: float = 1.0

    # Encoder architecture
    encoder: dict = field(default_factory=dict)

    # Regime/censoring
    pretrain_mask_current_day_to_open_only: bool = False
    augment_censor_last_timestep_prob: float = 0.0

    def fingerprint(self) -> str:
        """
        Stable-ish hash for quick equality checks.
        """
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()


def write_schema(path: Path, schema: SSLSchema) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(schema)
    data["fingerprint"] = schema.fingerprint()
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_schema(path: Path) -> SSLSchema:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fp = payload.pop("fingerprint", None)
    schema = SSLSchema(**payload)
    if fp is not None and str(fp) != schema.fingerprint():
        raise ValueError(f"schema fingerprint mismatch in {path}")
    return schema


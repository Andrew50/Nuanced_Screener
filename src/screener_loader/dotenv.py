from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DotenvResult:
    loaded: bool
    path: Path
    values: dict[str, str]


def parse_dotenv(text: str) -> dict[str, str]:
    """
    Minimal .env parser.
    Supports:
      - blank lines and # comments
      - KEY=VALUE (VALUE may be quoted)
      - export KEY=VALUE
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip()
        if not key:
            continue
        # Strip simple quotes
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        out[key] = val
    return out


def load_dotenv_file(path: Path) -> DotenvResult:
    if not path.exists():
        return DotenvResult(loaded=False, path=path, values={})
    text = path.read_text(encoding="utf-8")
    values = parse_dotenv(text)
    return DotenvResult(loaded=True, path=path, values=values)


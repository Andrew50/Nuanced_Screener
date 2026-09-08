from __future__ import annotations

from dataclasses import dataclass

from .spec import SetupSpec, VisionExample

_QUALITY_ORDER = {
    "canonical": 0,
    "decent": 1,
    "edge_case": 2,
    "near_miss": 3,
    None: 4,
}


@dataclass(frozen=True)
class CompiledPrompt:
    setup_id: str
    text: str
    example_ids: tuple[str, ...]


def compile_prompt(spec: SetupSpec, examples: list[VisionExample]) -> CompiledPrompt:
    ordered = _sort_examples(examples)
    lines: list[str] = [
        f"Setup: {spec.name} ({spec.id})",
        f"Timeframe: {spec.timeframe}",
        f"Lookback bars: {spec.lookback_bars}",
        "",
        "Description:",
        spec.description.strip() or "(none)",
        "",
        "Required:",
        *_bullet(spec.criteria.required),
        "",
        "Preferred:",
        *_bullet(spec.criteria.preferred),
        "",
        "Disqualifiers:",
        *_bullet(spec.criteria.disqualifiers),
    ]
    if spec.llm_notes.strip():
        lines.extend(["", "Notes:", spec.llm_notes.strip()])

    lines.extend(["", "Reference examples (few-shot):"])
    if not ordered:
        lines.append("- (none)")
    else:
        for ex in ordered:
            lines.append(f"- {_example_line(ex)}")

    text = "\n".join(lines).rstrip() + "\n"
    return CompiledPrompt(setup_id=spec.id, text=text, example_ids=tuple(ex.id for ex in ordered))


def _bullet(items: tuple[str, ...]) -> list[str]:
    if not items:
        return ["- (none)"]
    return [f"- {item}" for item in items]


def _sort_examples(examples: list[VisionExample]) -> list[VisionExample]:
    def key(ex: VisionExample) -> tuple[int, int, str]:
        pol = 0 if ex.polarity == "positive" else 1
        return (pol, _QUALITY_ORDER.get(ex.quality, 4), ex.id)

    return sorted(examples, key=key)


def _example_line(ex: VisionExample) -> str:
    quality = ex.quality or "unspecified"
    if ex.type == "market_window":
        loc = f"{ex.ticker} {ex.date.isoformat() if ex.date else ''} {ex.timeframe}".strip()
    else:
        loc = f"image {ex.path}"
    note = f" — {ex.note}" if ex.note else ""
    return f"{ex.polarity} [{quality}] {loc}{note}"

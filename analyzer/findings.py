"""The Finding record produced by every detector, plus ranking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
PRIORITY_LABEL = {"high": "P1", "medium": "P2", "low": "P3"}
_PREFIX = {"SEO": "SEO", "Ads": "ADS", "Cross-channel": "X", "Trend": "TREND"}


@dataclass
class Finding:
    area: str                 # "SEO" | "Ads" | "Cross-channel" | "Trend"
    kind: str                 # "issue" | "opportunity"
    title: str
    detail: str               # plain-English explanation with the actual numbers
    severity: str             # "high" | "medium" | "low"
    suggested_action: str     # deterministic first-pass action (the LLM refines it)
    value_usd: float | None = None   # estimated revenue at stake / upside; None = not quantified
    entities: list[dict[str, Any]] = field(default_factory=list)
    id: str = ""

    @property
    def priority(self) -> str:
        return PRIORITY_LABEL[self.severity]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "area": self.area, "kind": self.kind,
            "priority": self.priority, "severity": self.severity,
            "title": self.title, "detail": self.detail,
            "suggested_action": self.suggested_action,
            "value_usd": self.value_usd, "entities": self.entities,
        }


def rank_and_number(findings: list[Finding]) -> list[Finding]:
    """Sort by severity, then by estimated revenue at stake, and assign stable IDs."""
    ordered = sorted(
        findings,
        key=lambda f: (SEVERITY_ORDER[f.severity], -(f.value_usd or 0.0)),
    )
    counters: dict[str, int] = {}
    for f in ordered:
        prefix = _PREFIX.get(f.area, "F")
        counters[prefix] = counters.get(prefix, 0) + 1
        f.id = f"{prefix}-{counters[prefix]:02d}"
    return ordered

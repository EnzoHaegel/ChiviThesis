"""Risk lexicon — the seed knowledge used for weak-supervision labeling.

Phrases are grouped by risk category. They are used to:
  * generate ground-truth risk spans in documents (Phase 1 labeling), and
  * define risk-anchor tokens that seed candidate region proposals.

The lexicon is intentionally broad across the canonical 10-K/10-Q risk taxonomy
so it generalises across firms. Matching is case-insensitive and uses normalised
(lower-cased) token text. Multi-word phrases are matched greedily longest-first.
"""
from __future__ import annotations

from dataclasses import dataclass

# category -> list of canonical risk phrases (lower-case)
RISK_LEXICON: dict[str, list[str]] = {
    "macro_economic": [
        "economic recession", "recession", "economic downturn", "inflation",
        "rising interest rates", "interest rate", "interest rate risk",
        "monetary policy", "economic uncertainty", "gross domestic product",
        "unemployment", "consumer spending", "economic conditions", "stagflation",
        "deflation", "credit availability", "capital markets",
    ],
    "market": [
        "market volatility", "volatility", "stock price", "equity markets",
        "market conditions", "declining demand", "competitive pressure",
        "competition", "pricing pressure", "commodity prices", "foreign currency",
        "exchange rate", "currency fluctuations", "market share",
    ],
    "credit_liquidity": [
        "liquidity", "liquidity risk", "credit risk", "default", "indebtedness",
        "leverage", "debt obligations", "covenants", "refinancing", "downgrade",
        "credit rating", "cash flow", "funding", "counterparty risk", "insolvency",
        "going concern", "impairment", "goodwill impairment",
    ],
    "operational": [
        "supply chain", "supply chain disruption", "supplier", "manufacturing",
        "production", "labor shortage", "workforce", "key personnel",
        "operational disruption", "business interruption", "raw materials",
        "inventory", "distribution", "product recall", "quality control",
    ],
    "regulatory_legal": [
        "regulation", "regulatory", "litigation", "legal proceedings", "lawsuit",
        "compliance", "tax", "tax law", "antitrust", "intellectual property",
        "patent", "government investigation", "fines", "penalties", "sanctions",
        "environmental regulation", "data privacy regulation",
    ],
    "cyber_technology": [
        "cybersecurity", "cyber attack", "cyberattack", "data breach",
        "information security", "system failure", "ransomware", "hacking",
        "privacy", "technology disruption", "artificial intelligence",
        "software vulnerability", "network security", "data loss",
    ],
    "geopolitical_trade": [
        "tariff", "tariffs", "trade war", "trade policy", "trade restrictions",
        "geopolitical", "war", "armed conflict", "political instability",
        "sanctions", "import", "export controls", "national security",
        "russia", "ukraine", "china", "supply disruption",
    ],
    "health_pandemic": [
        "pandemic", "covid", "covid-19", "epidemic", "public health",
        "health crisis", "outbreak", "quarantine",
    ],
    "climate_esg": [
        "climate change", "climate", "extreme weather", "natural disaster",
        "environmental", "carbon emissions", "sustainability", "esg",
        "greenhouse gas", "physical risk", "transition risk", "flooding",
        "wildfire", "hurricane",
    ],
}


@dataclass(frozen=True)
class LexEntry:
    phrase: str
    tokens: tuple[str, ...]
    category: str


def build_entries() -> list[LexEntry]:
    """Flatten the lexicon into entries, deduplicated, sorted longest-first."""
    seen: dict[str, LexEntry] = {}
    for category, phrases in RISK_LEXICON.items():
        for phrase in phrases:
            toks = tuple(phrase.split())
            if phrase not in seen:
                seen[phrase] = LexEntry(phrase=phrase, tokens=toks, category=category)
    # Longest phrases first so greedy matching prefers the most specific term.
    return sorted(seen.values(), key=lambda e: (-len(e.tokens), e.phrase))


def anchor_tokens(entries: list[LexEntry] | None = None) -> set[str]:
    """Set of all tokens appearing in any lexicon phrase (region-proposal seeds)."""
    if entries is None:
        entries = build_entries()
    anchors: set[str] = set()
    for e in entries:
        anchors.update(e.tokens)
    return anchors


def category_of(phrase: str, entries: list[LexEntry] | None = None) -> str | None:
    if entries is None:
        entries = build_entries()
    for e in entries:
        if e.phrase == phrase:
            return e.category
    return None

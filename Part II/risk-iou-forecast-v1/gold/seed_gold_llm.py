"""LLM-annotated seed gold standard (independent of the risk lexicon).

Each entry maps a worksheet sent_uid -> the genuine risk-keyword / risk-driver
phrases present in that sentence, judged by reading the sentence (NOT by the
seed lexicon). Boilerplate sentences (forward-looking disclaimers, cross-
references to other risk-factor sections, share-repurchase notes, neutral
descriptive accounting) are left empty on purpose — they contain no specific
disclosed risk.

This is a *model-assisted* gold draft to demonstrate the validation loop and
give an immediate independent F1. For a thesis-grade number, a human (Enzo)
should review/adjust these labels in gold/worksheet.csv -> gold_annotations.csv.

Run:  python gold/seed_gold_llm.py   (writes gold/gold_annotations.csv)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# sent_uid -> ';'-separated gold risk phrases (must be contiguous token
# subsequences of the sentence; matching is case-insensitive).
GOLD: dict[int, str] = {
    0: "economic conditions; COVID-19",
    1: "",
    2: "risk of loss",
    3: "",
    4: "raw materials; tariffs; China",
    5: "",
    6: "cyber threats",
    7: "shortages; price escalation",
    8: "purchase commitments; obsolescence",
    9: "",
    10: "",
    11: "",
    12: "equity market decline; redemption requests",
    13: "legislation",
    14: "",
    15: "",
    16: "",
    17: "",
    18: "interest rate changes; demand",
    19: "",
    20: "",
    21: "natural gas prices; working capital",
    22: "",
    23: "",
    24: "personal information",
    25: "credit market volatility; credit ratings; liquidity",
    26: "material weakness; internal control over financial reporting",
    27: "federal debt ceiling; debt ceiling",
    28: "",
    29: "legal matters",
    30: "share price",
    31: "foreign countries",
    32: "financial privacy; anti-money laundering; corporate taxation; artificial intelligence",
    33: "pricing",
    34: "",
    35: "litigation proceedings",
    36: "investment adviser",
    37: "",
    38: "catastrophe exposure; litigation",
    39: "",
    40: "",
    41: "liquidity",
    42: "commodity prices",
    43: "Dodd-Frank Act; Consumer Financial Protection Bureau",
    44: "",
    45: "",
    46: "supply",
    47: "subscription",
    48: "",
    49: "GDPR",
    50: "",
    51: "",
    52: "",
    53: "cyber security; privacy risks; social engineering",
    54: "",
    55: "market rates",
    56: "",
    57: "taxation regulations; effective tax rate",
    58: "solvency; capital ratios; insurance regulators",
    59: "COVID-19 pandemic",
    60: "cybersecurity; personal information",
    61: "CARD Act",
    62: "commodities; trade disputes; tariffs; sanctions",
    63: "derivative instruments",
    64: "strike; union",
    65: "",
    66: "data privacy and security regulations",
    67: "securitization programs; AAA ratings",
    68: "",
    69: "",
    70: "natural disasters; hurricanes; earthquakes; severe weather",
    71: "",
    72: "lawsuits; governmental investigations; sanctions",
    73: "liquidity requirements",
    74: "",
    75: "",
    76: "changes in interest rates; net interest income",
    77: "uncertain tax position",
    78: "grounding",
    79: "",
    80: "",
    81: "",
    82: "LIFO method",
    83: "",
    84: "GHG",
    85: "",
    86: "collateral; unpaid loan",
    87: "suppliers; receivables",
    88: "collective bargaining agreements; unions",
    89: "FAA",
}


def main() -> int:
    ws_path = REPO / "gold" / "worksheet.csv"
    df = pd.read_csv(ws_path).fillna("")
    df["gold_risk_terms"] = df["sent_uid"].map(lambda u: GOLD.get(int(u), ""))
    out = REPO / "gold" / "gold_annotations.csv"
    df.to_csv(out, index=False)
    n_terms = sum(len([p for p in v.split(";") if p.strip()]) for v in GOLD.values())
    n_annotated = sum(1 for v in GOLD.values() if v.strip())
    print(f"wrote {out}")
    print(f"{n_annotated}/{len(df)} sentences carry >=1 gold risk term; "
          f"{n_terms} gold phrases total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

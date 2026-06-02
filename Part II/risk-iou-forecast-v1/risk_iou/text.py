"""Lightweight, dependency-free tokenisation.

We avoid nltk so the pipeline runs in any environment. Tokens carry their
character offsets so detected token spans can be mapped back to the source text
(needed for human-readable "pop out the words" output).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A token is a maximal run of word characters, apostrophes or internal hyphens.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

# Function words that should never sit on the boundary of a risk keyword span.
# Trimming them aligns candidate proposals with the (stopword-free) lexicon
# ground truth, which both raises detection precision and yields clean,
# human-readable "popped-out" risk terms.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but nor so yet for of to in on at by with from as into onto
    over under about above below between among through during before after since
    until per via is are was were be been being am do does did has have had having
    will would shall should can could may might must this that these those it its
    we our us you your they their he she his her i my me which who whom whose what
    not no any all some such other more most than then there here also however
    """.split()
)
# Sentence boundary: . ! ? optionally followed by quotes/brackets, then whitespace.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]?\s+")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Token:
    text: str        # original surface form
    norm: str        # lower-cased normalised form (used for matching)
    char_start: int
    char_end: int


def normalize(text: str) -> str:
    """Collapse whitespace; used before length-sensitive operations."""
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[Token]:
    """Tokenise `text`, preserving character offsets into the *original* string."""
    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(text):
        surface = m.group(0)
        tokens.append(
            Token(
                text=surface,
                norm=surface.lower(),
                char_start=m.start(),
                char_end=m.end(),
            )
        )
    return tokens


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter (regex on terminal punctuation)."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def ngrams(tokens: list[Token], n: int):
    """Yield (start_idx, end_idx, norm_text) for every contiguous n-gram."""
    for i in range(len(tokens) - n + 1):
        window = tokens[i : i + n]
        yield i, i + n, " ".join(t.norm for t in window)

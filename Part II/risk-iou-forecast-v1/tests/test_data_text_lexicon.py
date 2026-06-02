from risk_iou.data import parse_filename, _quarter
from risk_iou.text import tokenize, split_sentences, ngrams
from risk_iou.lexicon import build_entries, anchor_tokens, category_of


def test_parse_filename_valid():
    name = "2024-07-11_10-K_edgar_data_0000023217_0001558370-24-009764.txt"
    d = parse_filename(name)
    assert d is not None
    assert d["date"] == "2024-07-11"
    assert d["form"] == "10-K"
    assert d["cik"] == "0000023217"
    assert d["quarter"] == "2024Q3"


def test_parse_filename_10q():
    name = "2006-02-28_10-Q_edgar_data_0000006951_0000950134-06-003937.txt"
    d = parse_filename(name)
    assert d["form"] == "10-Q"
    assert d["quarter"] == "2006Q1"


def test_parse_filename_invalid():
    assert parse_filename("not_a_filing.txt") is None
    assert parse_filename("README.md") is None


def test_quarter_boundaries():
    assert _quarter("2023-01-15") == (2023, "2023Q1")
    assert _quarter("2023-03-31") == (2023, "2023Q1")
    assert _quarter("2023-04-01") == (2023, "2023Q2")
    assert _quarter("2023-12-31") == (2023, "2023Q4")


def test_tokenize_offsets():
    text = "Inflation and tariffs."
    toks = tokenize(text)
    assert [t.norm for t in toks] == ["inflation", "and", "tariffs"]
    # offsets map back to the original text
    for t in toks:
        assert text[t.char_start : t.char_end] == t.text


def test_split_sentences():
    s = split_sentences("Risk one is here. Risk two follows! Done?")
    assert len(s) == 3


def test_ngrams():
    toks = tokenize("supply chain disruption")
    grams = list(ngrams(toks, 2))
    assert grams[0][2] == "supply chain"
    assert grams[1][2] == "chain disruption"


def test_lexicon_entries_sorted_longest_first():
    entries = build_entries()
    lengths = [len(e.tokens) for e in entries]
    assert lengths == sorted(lengths, reverse=True)
    assert category_of("inflation", entries) == "macro_economic"


def test_anchor_tokens_nonempty():
    anchors = anchor_tokens()
    assert "inflation" in anchors
    assert "tariff" in anchors

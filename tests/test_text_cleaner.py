"""
Tests for utils/text_cleaner.py.

Three sections here exist specifically because a real bug was found
and fixed in text_cleaner during development, and each deserves a
permanent regression test so it can't silently reappear:

1. Stopword-removal phantom bigrams -- "saves so much time" must not
   produce "saves much" once "so" is dropped.
2. Punctuation-boundary phantom bigrams -- "interface & saves" must
   not produce "interface saves"; comma/ampersand/slash must break a
   phrase run the same way sentence punctuation does.
3. Redundant subsumed unigrams -- "crashes" (23), "constantly" (23),
   "crashes constantly" (23) must collapse to just the informative
   bigram, since equal counts prove they describe the same reviews.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from models.analysis import ThemeItem
from models.review import Review
from utils import text_cleaner


def _review(review_id: str, content: str, rating: int = 1) -> Review:
    return Review(review_id=review_id, rating=rating, content=content, review_date=datetime(2026, 7, 1))


# --- tokenize() --------------------------------------------------------------


def test_tokenize_lowercases_and_strips_stopwords():
    tokens = text_cleaner.tokenize("The App Crashes Constantly On My Phone")
    assert tokens == ["app", "crashes", "constantly", "phone"]


def test_tokenize_drops_short_tokens():
    tokens = text_cleaner.tokenize("it is ok I go")
    # "ok" (2 chars) and "go" (2 chars) both fall below the 3-char minimum;
    # "it", "is", "I" are stopwords/too short regardless.
    assert tokens == []


def test_tokenize_handles_empty_and_none_gracefully():
    assert text_cleaner.tokenize("") == []
    assert text_cleaner.tokenize(None) == []


def test_tokenize_strips_apostrophes_and_still_matches_stopwords():
    # "doesn't" should be recognized as a stopword even with the apostrophe.
    tokens = text_cleaner.tokenize("This app doesn't work well")
    assert "doesn't" not in tokens
    assert "work" in tokens and "well" in tokens


# --- _content_runs(): the two phantom-bigram bug fixes -----------------------


def test_content_runs_breaks_at_removed_stopwords():
    """Regression test: 'saves so much time' must not bridge 'saves' to 'much'."""
    runs = list(text_cleaner._content_runs("saves so much time"))
    assert ["saves"] in runs
    assert ["much", "time"] in runs
    # The phantom bigram this bug used to produce:
    joined_pairs = {" ".join(r) for r in runs}
    assert "saves much" not in joined_pairs


def test_content_runs_breaks_at_ampersand_and_comma():
    """Regression test: punctuation-like separators must break a run too, not just stopwords."""
    runs = list(text_cleaner._content_runs("Clean interface & saves so much time"))
    assert ["clean", "interface"] in runs
    assert ["saves"] in runs
    assert ["much", "time"] in runs
    joined_pairs = {" ".join(r) for r in runs}
    assert "interface saves" not in joined_pairs


def test_content_runs_breaks_at_sentence_punctuation():
    runs = list(text_cleaner._content_runs("Great app. Crashes constantly though."))
    joined_pairs = {" ".join(r) for r in runs}
    assert "app crashes" not in joined_pairs


# --- extract_common_themes(): document-frequency counting --------------------


def test_extract_common_themes_counts_reviews_not_occurrences():
    """A phrase repeated many times within ONE review must not outrank a phrase in several reviews."""
    reviews = [
        _review("r1", "crashes crashes crashes crashes crashes"),  # 1 review, word repeated 5x
        _review("r2", "battery drain issue"),
        _review("r3", "battery drain issue"),
    ]
    themes = text_cleaner.extract_common_themes(reviews, min_reviews=2)
    phrases = {t.phrase: t.count for t in themes}
    assert "crashes" not in phrases  # only appears in 1 distinct review, below min_reviews=2
    assert phrases.get("battery drain") == 2


def test_extract_common_themes_respects_min_reviews_threshold():
    reviews = [_review("r1", "unique complaint about sync issues")]
    themes = text_cleaner.extract_common_themes(reviews, min_reviews=2)
    assert themes == []  # only 1 review mentions it -- below threshold, correctly excluded


def test_extract_common_themes_respects_top_n():
    # Build 5 distinct 2-review-strength themes; top_n=2 must return only the 2 most frequent.
    reviews = []
    for i, phrase in enumerate(["alpha", "bravo", "charlie", "delta", "echo"]):
        count = 5 - i  # alpha appears most, echo least
        for j in range(count):
            reviews.append(_review(f"r{i}-{j}", phrase))
    themes = text_cleaner.extract_common_themes(reviews, ngram_sizes=(1,), min_reviews=2, top_n=2)
    assert len(themes) == 2
    assert themes[0].phrase == "alpha"
    assert themes[1].phrase == "bravo"


def test_extract_common_themes_handles_empty_and_blank_content():
    reviews = [_review("r1", ""), _review("r2", "   "), _review("r3", "real complaint text here")]
    themes = text_cleaner.extract_common_themes(reviews, min_reviews=1)
    # Doesn't crash on blank/empty content, and the one real review still contributes.
    # (Whether the surviving phrase is the unigram "real" or the bigram "real complaint"
    # depends on subsumed-unigram dedup -- either is correct here, so match loosely.)
    assert len(themes) > 0
    assert any("real" in t.phrase for t in themes)


# --- Regression test: subsumed-unigram dedup ----------------------------------


def test_dedupe_drops_unigram_when_bigram_has_identical_count():
    """Regression test: 'crashes'(23) + 'constantly'(23) + 'crashes constantly'(23) -> just the bigram."""
    reviews = [_review(f"r{i}", "The app crashes constantly and it is so annoying") for i in range(23)]
    themes = text_cleaner.extract_common_themes(reviews, min_reviews=2)
    phrases = {t.phrase for t in themes}
    assert "crashes constantly" in phrases
    assert "crashes" not in phrases
    assert "constantly" not in phrases


def test_dedupe_keeps_unigram_when_counts_differ():
    """A unigram with mentions BEYOND any single bigram's reach must survive -- it's not fully subsumed."""
    # "crashes" is adjacent to "constantly" in one group and to different words in
    # another, so no single bigram ever reaches "crashes"'s full count of 13.
    reviews = [_review(f"r{i}", "the app crashes constantly here") for i in range(10)]
    reviews += [_review(f"extra{i}", "sadly it still crashes sometimes") for i in range(3)]
    themes = text_cleaner.extract_common_themes(reviews, min_reviews=2)
    phrases = {t.phrase: t.count for t in themes}
    assert phrases.get("crashes") == 13
    assert phrases.get("crashes constantly") == 10


def test_dedupe_only_matches_whole_words_not_substrings():
    """
    'cat' must not be treated as subsumed by 'category error' just
    because 'cat' is a SUBSTRING of 'category' -- the dedup check
    must compare whole words (via .split()), not raw string
    containment, or unrelated words would wrongly collapse together.
    """
    themes = [
        ThemeItem(phrase="category error", count=5),
        ThemeItem(phrase="cat", count=5),
    ]
    deduped = text_cleaner._dedupe_subsumed_unigrams(themes)
    # A naive substring check ("cat" in "category error") would wrongly drop "cat" here.
    assert deduped == themes


# --- pick_lead_theme(): multi-word preference ---------------------------------


def test_pick_lead_theme_prefers_multiword_over_higher_ranked_unigram():
    themes = [
        ThemeItem(phrase="clean", count=40),  # ranked first by count, but generic
        ThemeItem(phrase="clean interface", count=35),
    ]
    lead = text_cleaner.pick_lead_theme(themes)
    assert lead.phrase == "clean interface"


def test_pick_lead_theme_falls_back_to_unigram_when_no_multiword_exists():
    themes = [ThemeItem(phrase="buggy", count=10), ThemeItem(phrase="slow", count=8)]
    lead = text_cleaner.pick_lead_theme(themes)
    assert lead.phrase == "buggy"


def test_pick_lead_theme_handles_empty_list():
    assert text_cleaner.pick_lead_theme([]) is None


# --- clean_text() --------------------------------------------------------------


def test_clean_text_collapses_whitespace():
    assert text_cleaner.clean_text("  too   much   space \n here  ") == "too much space here"


def test_clean_text_handles_none():
    assert text_cleaner.clean_text(None) == ""


# --- extract_common_themes(): exclude_terms (target app name exclusion) -----


def _app_name_reviews() -> list[Review]:
    """
    Deliberately shaped so 'whatsapp' the unigram would normally
    SURVIVE the pre-existing subsumed-unigram dedup on its own (no
    single bigram shares its exact count of 5 -- 3 from one phrasing,
    2 from another) -- so excluding it in the tests below can be
    cleanly attributed to the new exclude_terms feature, not to dedup
    already removing it for an unrelated reason.
    """
    reviews = [_review(f"r{i}", "whatsapp crashes on startup") for i in range(3)]
    reviews += [_review(f"x{i}", "the whatsapp app is annoying today") for i in range(2)]
    return reviews


def test_extract_common_themes_excludes_given_unigram_terms():
    """The target app's own name should not itself count as a complaint/praise theme."""
    themes = text_cleaner.extract_common_themes(
        _app_name_reviews(), min_reviews=2, exclude_terms=frozenset({"whatsapp"})
    )
    phrases = {t.phrase for t in themes}
    assert "whatsapp" not in phrases


def test_extract_common_themes_retains_multiword_phrase_containing_excluded_term():
    """A bigram like 'whatsapp crashes' carries real signal the bare mention doesn't -- must survive."""
    themes = text_cleaner.extract_common_themes(
        _app_name_reviews(), min_reviews=2, exclude_terms=frozenset({"whatsapp"})
    )
    phrases = {t.phrase for t in themes}
    assert "whatsapp crashes" in phrases
    assert "whatsapp app" in phrases


def test_extract_common_themes_default_exclude_terms_is_empty_and_unchanged():
    """Calling without exclude_terms must behave exactly as before this feature existed."""
    reviews = _app_name_reviews()
    with_default = text_cleaner.extract_common_themes(reviews, min_reviews=2)
    with_explicit_empty = text_cleaner.extract_common_themes(reviews, min_reviews=2, exclude_terms=frozenset())
    assert with_default == with_explicit_empty
    phrases = {t.phrase for t in with_default}
    assert "whatsapp" in phrases  # nothing excluded when exclude_terms is empty/omitted


def test_extract_common_themes_does_not_filter_generic_words_by_default():
    """
    The generic-word policy (settings.GENERIC_THEME_TERMS) lives in
    analysis_service.analyze(), not inside extract_common_themes
    itself -- calling the utility directly without exclude_terms must
    still allow a word like 'app' as a standalone theme candidate.
    Fixture shaped so 'app' appears in two different adjacency
    contexts (count=5 total, no single bigram shares that count), so
    it would survive the pre-existing dedup on its own too -- its
    presence here is unambiguous proof of no built-in filtering.
    """
    reviews = [_review(f"r{i}", "app crashes on startup") for i in range(3)]
    reviews += [_review(f"x{i}", "the app is annoying today") for i in range(2)]
    themes = text_cleaner.extract_common_themes(reviews, min_reviews=2)
    phrases = {t.phrase for t in themes}
    assert "app" in phrases

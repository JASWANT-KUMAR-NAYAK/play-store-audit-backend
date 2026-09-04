"""
Text cleaning and deterministic keyword/phrase (theme) extraction.

No ML, no external NLP models -- just regex tokenization, a
hardcoded English stopword list, and frequency counting, per the
locked V1 scope (collections.Counter / re / datetime only).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Iterable, Iterator, Optional

from config import settings
from models.analysis import ThemeItem

if TYPE_CHECKING:
    from models.review import Review

_WORD_PATTERN = re.compile(r"[a-z']+")
# Breaks a phrase run at sentence punctuation AND at clause-separating
# characters (comma, ampersand, slash) that aren't themselves letters
# and so would otherwise be silently skipped by _WORD_PATTERN -- e.g.
# without ',' and '&' here, "clean interface & saves so much time"
# lets "interface" and "saves" become falsely adjacent once "so" is
# removed as a stopword, producing a phantom bigram that never
# appeared in the original sentence.
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?;:()\[\]\n,&/]+")
_MIN_TOKEN_LEN = 3

# Standard English stopwords. Deliberately NOT domain-tuned (no
# app-review-specific exclusions like "app" or "please") -- the spec
# calls for common stopword removal, not hand-curated theme steering.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll
    he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our
    ours ourselves out over own same shan't she she'd she'll she's should
    shouldn't so some such than that that's the their theirs them
    themselves then there there's these they they'd they'll they're
    they've this those through to too under until up very was wasn't we
    we'd we'll we're we've were weren't what what's when when's where
    where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself
    yourselves
    """.split()
)


def clean_text(text: str) -> str:
    """Collapse whitespace and strip. Cheap normalization for display/logging."""
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    """
    Lowercase, extract alphabetic tokens, drop stopwords and short tokens.

    Flat token list (stopwords removed, order preserved) -- used for
    unigram counting where run-adjacency doesn't matter. For n-gram
    (n>=2) extraction, see `_content_runs`, which avoids stitching
    together words that were only adjacent in the *filtered* stream,
    not in the original text.
    """
    if not text:
        return []
    tokens: list[str] = []
    for raw in _WORD_PATTERN.findall(text.lower()):
        word = raw.strip("'")
        if len(word) < _MIN_TOKEN_LEN:
            continue
        if word in STOPWORDS:
            continue
        tokens.append(word)
    return tokens


def _content_runs(text: str) -> Iterator[list[str]]:
    """
    Yield contiguous runs of content words, broken at BOTH punctuation
    and stopwords.

    This is what n-gram (n>=2) extraction should walk, not the flat
    stopword-stripped token list: removing "so" from "saves so much
    time" must not let "saves" and "much" become adjacent -- that
    produces a phantom bigram ("saves much") that never appeared in
    the original review. Breaking the run there instead correctly
    yields the separate, real bigram "much time".
    """
    # Punctuation/sentence boundaries first, so a bigram can never
    # straddle two different sentences.
    for fragment in _SENTENCE_SPLIT_PATTERN.split(text.lower()):
        current: list[str] = []
        for raw in _WORD_PATTERN.findall(fragment):
            word = raw.strip("'")
            is_content_word = len(word) >= _MIN_TOKEN_LEN and word not in STOPWORDS
            if is_content_word:
                current.append(word)
            elif current:
                yield current
                current = []
        if current:
            yield current


def _ngrams(tokens: list[str], n: int) -> Iterator[str]:
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def pick_lead_theme(themes: list[ThemeItem]) -> Optional[ThemeItem]:
    """
    Pick the theme to feature in prose (executive summary narrative,
    recommendation sentences) -- as opposed to the full frequency-
    ranked table shown in the report.

    A generic unigram like "constantly" or "clean" makes for a weak,
    non-actionable headline. Prefer the highest-ranked multi-word
    phrase -- still drawn from the real frequency ranking, just more
    specific -- and only fall back to a unigram if no multi-word
    theme cleared the min_reviews threshold.
    """
    for theme in themes:
        if " " in theme.phrase:
            return theme
    return themes[0] if themes else None


def _dedupe_subsumed_unigrams(themes: list[ThemeItem]) -> list[ThemeItem]:
    """
    Drop a unigram when a multi-word theme containing it has the
    IDENTICAL count.

    A bigram's review-set is always a subset of each component word's
    review-set (a review can only contain "crashes constantly" if it
    contains "crashes"). When the counts match exactly, that subset
    relationship becomes set equality -- the unigram is describing the
    exact same reviews as the more specific phrase, so keeping both
    just repeats a row for no analytical gain (e.g. "crashes" (23),
    "constantly" (23), "crashes constantly" (23) collapses to just
    the one informative row).
    """
    multiword = [t for t in themes if " " in t.phrase]
    deduped: list[ThemeItem] = []
    for theme in themes:
        if " " in theme.phrase:
            deduped.append(theme)
            continue
        subsumed = any(
            theme.phrase in mw.phrase.split() and mw.count == theme.count for mw in multiword
        )
        if not subsumed:
            deduped.append(theme)
    return deduped


def extract_common_themes(
    reviews: Iterable["Review"],
    ngram_sizes: tuple[int, ...] = settings.THEME_NGRAM_SIZES,
    top_n: int = settings.THEME_TOP_N,
    min_reviews: int = 2,
) -> list[ThemeItem]:
    """
    Find recurring keyword/phrase themes across a set of reviews.

    Counts DOCUMENT frequency, not raw term frequency: each phrase is
    counted at most once per review, so a single review repeating a
    word can't masquerade as a widespread theme. Only phrases
    appearing in at least `min_reviews` distinct reviews are kept, so
    one-off phrasing doesn't clutter the results.

    Unigrams (n=1) are drawn from the flat stopword-stripped token
    list. n-grams of 2+ are drawn from `_content_runs`, so a bigram
    only appears if those two words were genuinely adjacent (modulo
    stopwords) in the original sentence.
    """
    counter: Counter[str] = Counter()

    for review in reviews:
        phrases_in_this_review: set[str] = set()

        if 1 in ngram_sizes:
            phrases_in_this_review.update(tokenize(review.content))

        multi_word_sizes = [n for n in ngram_sizes if n >= 2]
        if multi_word_sizes:
            for run in _content_runs(review.content):
                for n in multi_word_sizes:
                    phrases_in_this_review.update(_ngrams(run, n))

        counter.update(phrases_in_this_review)

    themes = [
        ThemeItem(phrase=phrase, count=count)
        for phrase, count in counter.most_common()
        if count >= min_reviews
    ]
    themes = _dedupe_subsumed_unigrams(themes)
    return themes[:top_n]

"""Build the retrieval golden set by TREC-style pooling and LLM judging.

WHY THIS EXISTS
---------------
The hand-written 2.0.0 set cannot gate a search backend change. Its queries
average 8.7 words (real users type 2-4), and it is saturated: semantic scores
MRR@10 0.96 and BM25 0.91, leaving no headroom to separate an improvement from
noise. It also does not scale -- every new country and source needs new
judgments, and hand-writing them does not survive 2 countries becoming 10.

METHOD
------
Standard pooling, as TREC uses for collections too large to judge exhaustively:

  1. Select topically DIVERSE articles (farthest-first traversal in embedding
     space), so queries spread across what the corpus actually covers.
  2. Have the judge write short, realistic queries for each selected article.
  3. Deduplicate queries semantically, not by string equality.
  4. Pool the union of every retrieval method's top results. Anything no
     method retrieves is assumed irrelevant.
  5. Judge each pooled (query, article) pair on a 0-3 scale.
  6. Drop queries that cannot discriminate between backends.
  7. Calibrate the judge against the human judgments and record the agreement.

Step 5 produces hard negatives for free: a pooled candidate graded 0 is, by
construction, something a real method ranked highly and a judge rejected.

KNOWN BIAS
----------
Pooling only judges what some method retrieved, so recall is measured against
the pool, not the corpus. This flatters the methods that built the pool and
penalises a future method that finds articles none of today's methods surface.
The pool must therefore be rebuilt when a materially different backend (such
as Qdrant BM25) is introduced -- which is also why BM25 is pooled here even
though it is not yet a production code path.

The same bias applies to hard negatives, and more sharply: a grade-0 candidate
is inside the contributing method's top results by construction, so
``false_positive_rate`` is NOT comparable across backends until every backend
contributes to the pool. It is recorded, but must not be used as a gate before
the rebuild.

Usage:
    PYTHONPATH=. .venv/bin/python evals/build_search_golden.py --articles 40
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy
from pydantic import BaseModel, Field
from sqlalchemy import select, text

from evals.util.eval_service import EvalService
from shared.database import SessionLocal
from shared.models import Articles, Countries, Sources
from shared.search_service import SearchService


EVAL_DIR = Path(__file__).resolve().parent
EXISTING_GOLDEN = EVAL_DIR / "golden" / "search.json"
DEFAULT_OUTPUT = EVAL_DIR / "golden" / "search-v3.json"

# How many pooled candidates each method contributes. Deep enough to catch the
# near-misses that become hard negatives, shallow enough to stay affordable.
POOL_DEPTH = 12

# Two queries whose embeddings are this close are the same question asked
# twice. String dedup does not catch "US Iran tensions" vs "Iran US tensions";
# this does.
DUPLICATE_SIMILARITY = 0.90

# A query relevant to more than this share of the corpus cannot discriminate:
# almost anything returned is a hit, so recall approaches 1.0 for every
# backend and the query measures nothing. Expressed as a fraction so the rule
# survives the corpus growing.
MAX_RELEVANT_FRACTION = 0.07

# BM25 parameters. Standard defaults, and what Qdrant uses, so the pooled BM25
# results approximate what the Qdrant backend will retrieve.
BM25_K1 = 1.2
BM25_B = 0.75

GRADE_GUIDE = (
    "3 = directly answers or is centrally about the query. "
    "2 = substantially covers the topic from another angle or as a major "
    "secondary element. "
    "1 = related context that mentions the topic but is about something "
    "else. "
    "0 = not relevant, including articles that merely share vocabulary, "
    "share a region, or cover a different event of the same kind."
)


class GeneratedQuery(BaseModel):
    query: str = Field(description="a short search query, 2-4 words")
    category: str = Field(description="one of: entity_lookup, event_lookup, "
                                      "topic_browse, detail_lookup")


class GeneratedQueries(BaseModel):
    queries: list[GeneratedQuery] = Field(description="the generated queries")


class CandidateGrade(BaseModel):
    index: int = Field(description="the candidate's number in the input list")
    grade: int = Field(description="relevance grade from 0 to 3")


class PoolGrades(BaseModel):
    grades: list[CandidateGrade] = Field(description="one grade per candidate")


class BM25Retriever:
    """A BM25 retriever used only to widen the judging pool.

    Tokenisation is delegated to Postgres's ``to_tsvector`` so the stemmer and
    stopword list match the existing lexical arm exactly. This exists because
    the production lexical arm uses AND semantics and returns nothing for most
    queries, which would leave the pool almost entirely semantic -- and a pool
    built by one method cannot fairly judge another.
    """

    def __init__(self) -> None:
        statement = text(
            """
            select a.link, t.lexeme, array_length(t.positions, 1)
            from articles a,
                 lateral unnest(
                     to_tsvector(
                         'english',
                         a.translated_headline || ' ' || a.summary
                     )
                 ) as t
            """
        )

        self.terms: dict[str, dict[str, int]] = {}
        with SessionLocal() as session:
            for link, lexeme, frequency in session.execute(statement):
                document = self.terms.setdefault(link, {})
                document[lexeme] = document.get(lexeme, 0) + (frequency or 1)

        self.lengths = {
            link: sum(terms.values()) for link, terms in self.terms.items()
        }
        self.document_count = len(self.terms) or 1
        self.average_length = (
            sum(self.lengths.values()) / self.document_count
        )
        self.document_frequency: Counter[str] = Counter()
        for terms in self.terms.values():
            self.document_frequency.update(terms.keys())

    def search(self, query: str, limit: int = POOL_DEPTH) -> list[str]:
        """
        Ranks the corpus by BM25 and returns the top links.

        :param query: the search query
        :param limit: how many links to return
        :return: the highest scoring article links
        """

        with SessionLocal() as session:
            query_terms = [
                row[0] for row in session.execute(
                    text("select lexeme from unnest("
                         "to_tsvector('english', :query))"),
                    {"query": query},
                )
            ]

        scores: dict[str, float] = {}
        for link, terms in self.terms.items():
            total = 0.0
            for term in query_terms:
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(
                    1 + (self.document_count - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + BM25_K1 * (
                    1 - BM25_B
                    + BM25_B * (self.lengths[link] / self.average_length)
                )
                total += idf * (frequency * (BM25_K1 + 1) / denominator)
            if total > 0:
                scores[link] = total

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return [link for link, _ in ranked[:limit]]


def select_diverse_articles(count: int, seed: int) -> list[Articles]:
    """
    Selects topically spread articles by farthest-first traversal over their
    embeddings.

    Sampling per source does NOT work here: all six outlets cover the same
    stories, so source-stratified sampling yields source diversity and topic
    monoculture -- which is how an earlier build produced 31 near-duplicate
    query pairs. Choosing each next article to be maximally distant from
    everything already chosen spreads the selection across what the corpus
    actually covers, and keeps working as more countries are added.

    :param count: how many articles to select
    :param seed: the random seed for the starting point, so builds reproduce
    :return: the selected article objects
    """

    with SessionLocal() as session:
        rows = session.execute(
            select(Articles).where(Articles.embedding.is_not(None))
        ).scalars().all()

    if len(rows) <= count:
        return list(rows)

    vectors = numpy.array([row.embedding for row in rows], dtype=numpy.float32)
    vectors /= numpy.linalg.norm(vectors, axis=1, keepdims=True)

    start = random.Random(seed).randrange(len(rows))
    chosen = [start]
    # Distance from every article to the nearest already-chosen article.
    nearest = 1.0 - vectors @ vectors[start]

    while len(chosen) < count:
        candidate = int(numpy.argmax(nearest))
        chosen.append(candidate)
        nearest = numpy.minimum(nearest, 1.0 - vectors @ vectors[candidate])
        nearest[candidate] = -1.0

    return [rows[index] for index in chosen]


class GoldenSetBuilder:
    def __init__(self, model: str = "gpt-5.4-mini"):
        self.judge = EvalService(model=model)
        self.search = SearchService()
        self.bm25 = BM25Retriever()
        self.articles_by_link: dict[str, Articles] = {}

    def load_article_index(self) -> None:
        """Caches every article by link so pooled links can be hydrated."""

        with SessionLocal() as session:
            for article in session.execute(select(Articles)).scalars():
                self.articles_by_link.setdefault(article.link, article)

    def write_queries(self, article: Articles) -> list[GeneratedQuery]:
        """
        Asks the judge for short queries a real user would type to find this
        article. Length is the whole point: the 2.0.0 set wrote queries as
        full sentences, which no search box ever receives.

        :param article: the article to derive queries from
        :return: the generated queries, filtered to 2-4 words
        """

        response = self.judge.client.responses.parse(
            model=self.judge.model,
            instructions="You write realistic search queries for a news "
                         "archive covering multiple countries and outlets. "
                         "Given an article, write 2 queries a real person "
                         "would type into a search box to find it. Queries "
                         "MUST be 2-4 words. Never write a full sentence. "
                         "Use words a user would know BEFORE reading the "
                         "article: place names, people, organisations, plain "
                         "topic words. Do not copy the headline. Both queries "
                         "must be SPECIFIC enough that only a handful of "
                         "articles could answer them. Do not write generic "
                         "geopolitical queries such as 'Iran tensions' or "
                         "'US Iran talks' -- those match half the archive and "
                         "are useless. Prefer named people, named places, "
                         "named institutions, and specific incidents.",
            input="Headline:\n" + (article.translated_headline or "") +
                  "\n\nSummary:\n" + (article.summary or "")[:1500],
            text_format=GeneratedQueries,
        )

        parsed = response.output_parsed
        return [
            candidate for candidate in (parsed.queries if parsed else [])
            if 2 <= len(candidate.query.split()) <= 4
        ]

    def pool_candidates(self, query: str) -> list[Articles]:
        """
        Unions the top results of every retrieval method, including BM25.

        Judging the union rather than one method's output is what keeps the
        golden set from encoding the biases of whichever backend built it.

        :param query: the query to pool candidates for
        :return: the deduplicated candidate articles
        """

        pooled: dict[str, Articles] = {}

        for method in (self.search.lexical_search,
                       self.search.semantic_search):
            try:
                for article, _ in method(query)[:POOL_DEPTH]:
                    pooled.setdefault(article.link, article)
            except Exception:
                continue

        for link in self.bm25.search(query, POOL_DEPTH):
            article = self.articles_by_link.get(link)
            if article is not None:
                pooled.setdefault(link, article)

        return list(pooled.values())

    def grade_pool(self, query: str,
                   candidates: list[Articles]) -> dict[str, int]:
        """
        Grades every pooled candidate against the query in a single call, so
        the judge sees candidates side by side and grades them relative to
        one another rather than in isolation.

        :param query: the search query being judged
        :param candidates: the pooled candidate articles
        :return: a mapping of article link to relevance grade
        """

        listing = "\n\n".join(
            f"[{index}] {article.translated_headline}\n"
            f"{(article.summary or '')[:600]}"
            for index, article in enumerate(candidates)
        )

        response = self.judge.client.responses.parse(
            model=self.judge.model,
            instructions="You are grading search results for a multilingual "
                         "news archive. Grade how well each candidate answers "
                         "the query. " + GRADE_GUIDE + " Be strict. A "
                         "candidate that shares a region, a conflict, a "
                         "country, or general vocabulary with the query but "
                         "reports a DIFFERENT event is 0, not 1. Most "
                         "candidates in a pool are usually 0. Reserve 3 for "
                         "articles genuinely about the query. Return exactly "
                         "one grade for every candidate index in the input.",
            input=f"Query: {query}\n\nCandidates:\n{listing}",
            text_format=PoolGrades,
        )

        parsed = response.output_parsed
        graded: dict[str, int] = {}
        for entry in (parsed.grades if parsed else []):
            if 0 <= entry.index < len(candidates) and 0 <= entry.grade <= 3:
                graded[candidates[entry.index].link] = entry.grade
        return graded

    def deduplicate(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Removes queries that repeat an earlier query's meaning.

        Exact string matching is not enough: "US Iran tensions" and "Iran US
        tensions" are the same question with the words reordered. Sorted-token
        equality catches reorderings and embedding similarity catches
        rephrasings.

        :param records: the generated query records, in creation order
        :return: the records with semantic duplicates removed
        """

        kept: list[dict[str, Any]] = []
        kept_vectors: list[numpy.ndarray] = []
        seen_tokens: set[frozenset] = set()

        for record in records:
            tokens = frozenset(record["query"].lower().split())
            if tokens in seen_tokens:
                continue

            vector = numpy.array(
                self.search.embed(record["query"]), dtype=numpy.float32
            )
            vector /= numpy.linalg.norm(vector)
            if kept_vectors:
                similarity = float(numpy.max(numpy.stack(kept_vectors) @ vector))
                if similarity >= DUPLICATE_SIMILARITY:
                    continue

            seen_tokens.add(tokens)
            kept.append(record)
            kept_vectors.append(vector)

        return kept

    def calibrate(self, human_queries: list[dict[str, Any]],
                  sample: int) -> dict[str, Any]:
        """
        Measures whether the LLM judge agrees with the human judgments.

        Without this the generated half of the set is unfalsifiable: there is
        no way to know whether a machine grade of 3 means what a human grade
        of 3 means. Regrades the human-judged pairs and reports exact
        agreement, agreement within one grade, and -- the number that actually
        matters for retrieval metrics -- agreement on the binary
        relevant/not-relevant split that every metric thresholds on.

        :param human_queries: the hand-judged query records
        :param sample: how many human queries to regrade
        :return: the agreement statistics
        """

        chosen = human_queries[:sample]
        pairs: list[tuple[int, int]] = []

        for record in chosen:
            candidates = [
                self.articles_by_link[link]
                for link in record["relevance"]
                if link in self.articles_by_link
            ]
            if not candidates:
                continue
            try:
                grades = self.grade_pool(record["query"], candidates)
            except Exception:
                continue
            for link, human_grade in record["relevance"].items():
                if link in grades:
                    pairs.append((human_grade, grades[link]))

        if not pairs:
            return {"pairs": 0}

        exact = sum(1 for human, llm in pairs if human == llm)
        within_one = sum(1 for human, llm in pairs if abs(human - llm) <= 1)
        binary = sum(
            1 for human, llm in pairs if (human > 0) == (llm > 0)
        )
        return {
            "pairs": len(pairs),
            "queries_regraded": len(chosen),
            "exact_agreement": round(exact / len(pairs), 4),
            "within_one_grade": round(within_one / len(pairs), 4),
            "binary_relevance_agreement": round(binary / len(pairs), 4),
            "mean_human_grade": round(
                sum(human for human, _ in pairs) / len(pairs), 3
            ),
            "mean_llm_grade": round(
                sum(llm for _, llm in pairs) / len(pairs), 3
            ),
        }


def load_existing(path: Path) -> list[dict[str, Any]]:
    """
    Carries the hand-written 2.0.0 queries forward verbatim, tagged with their
    provenance. They are the only human judgments in the set.

    :param path: the existing golden set file
    :return: the existing query records, tagged as human-judged
    """

    if not path.exists():
        return []

    document = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    for record in document.get("queries", []):
        record = dict(record)
        record["judged_by"] = "human"
        queries.append(record)
    return queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=int, default=40,
                        help="how many articles to seed queries from")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--calibration-sample", type=int, default=20,
                        help="human queries to regrade for judge agreement")
    parser.add_argument("--skip-existing", action="store_true",
                        help="omit the 2.0.0 human-judged queries")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = GoldenSetBuilder(model=args.model)
    builder.load_article_index()
    corpus_size = len(builder.articles_by_link)
    max_relevant = max(2, int(corpus_size * MAX_RELEVANT_FRACTION))

    articles = select_diverse_articles(args.articles, args.seed)
    print(f"Selected {len(articles)} topically diverse articles "
          f"from {corpus_size}.")
    print(f"Discriminative cap: a query may have at most {max_relevant} "
          f"relevant articles ({MAX_RELEVANT_FRACTION:.0%} of corpus).\n")

    drafts: list[dict[str, Any]] = []
    for position, article in enumerate(articles, start=1):
        try:
            for candidate in builder.write_queries(article):
                drafts.append({
                    "query": candidate.query.strip(),
                    "category": candidate.category,
                    "seed_article": article.link,
                })
        except Exception as exception:
            print(f"  [{position}] query generation failed: {exception}")

    print(f"Generated {len(drafts)} candidate queries.")
    drafts = builder.deduplicate(drafts)
    print(f"After semantic deduplication: {len(drafts)}.\n")

    generated: list[dict[str, Any]] = []
    dropped_broad = 0
    dropped_empty = 0

    for draft in drafts:
        pooled = builder.pool_candidates(draft["query"])
        if not pooled:
            dropped_empty += 1
            continue

        try:
            grades = builder.grade_pool(draft["query"], pooled)
        except Exception as exception:
            print(f"  grading failed for {draft['query']!r}: {exception}")
            continue

        positives = sum(1 for grade in grades.values() if grade > 0)
        if not positives:
            dropped_empty += 1
            continue
        if positives > max_relevant:
            # Non-discriminative: nearly anything returned is a hit, so every
            # backend scores ~1.0 and the query separates nothing.
            dropped_broad += 1
            print(f"  drop {draft['query']!r}: {positives} relevant "
                  f"> cap {max_relevant}")
            continue

        generated.append({
            "id": f"search-v3-{len(generated) + 1:03d}",
            "query": draft["query"],
            "category": draft["category"],
            "judged_by": "llm",
            "seed_article": draft["seed_article"],
            "relevance": grades,
        })
        negatives = sum(1 for grade in grades.values() if grade == 0)
        print(f"  [{len(generated):03d}] {draft['query']!r} "
              f"-> {positives} relevant, {negatives} hard negatives")

    human = [] if args.skip_existing else load_existing(EXISTING_GOLDEN)

    print("\nCalibrating judge against human judgments...")
    calibration = builder.calibrate(
        load_existing(EXISTING_GOLDEN), args.calibration_sample
    )
    print(f"  {calibration}")

    queries = human + generated
    document = {
        "version": "3.0.0",
        "description": (
            "Retrieval judgments for the OsiPress corpus. Combines the "
            "hand-written 2.0.0 queries (judged_by: human) with short, "
            "realistic queries seeded from topically diverse articles, "
            "deduplicated by embedding similarity, pooled across lexical, "
            "semantic and BM25 retrieval, and graded by an LLM judge "
            "(judged_by: llm). Grade 3 = directly answers or is centrally "
            "about the query; 2 = substantially covers the topic from "
            "another angle; 1 = related context; 0 = a HARD NEGATIVE, "
            "retrieved by some method but judged irrelevant. Queries "
            "relevant to more than "
            f"{MAX_RELEVANT_FRACTION:.0%} of the corpus were dropped as "
            "non-discriminative. Recall is measured against the judged pool, "
            "not the full corpus, so the pool must be rebuilt when a "
            "materially different backend is introduced. For the same reason "
            "false_positive_rate is recorded but must NOT gate a backend "
            "that did not contribute to the pool."
        ),
        "build": {
            "seed": args.seed,
            "articles_selected": len(articles),
            "corpus_size": corpus_size,
            "selection": "farthest-first traversal over embeddings",
            "pool_depth": POOL_DEPTH,
            "pooled_methods": ["lexical", "semantic", "bm25"],
            "judge_model": args.model,
            "duplicate_similarity": DUPLICATE_SIMILARITY,
            "max_relevant_fraction": MAX_RELEVANT_FRACTION,
            "max_relevant_articles": max_relevant,
            "dropped_non_discriminative": dropped_broad,
            "dropped_no_relevant": dropped_empty,
            "judge_calibration": calibration,
        },
        "thresholds": {
            "mrr_at_10": 0.8,
            "ndcg_at_10": 0.75,
            "recall_at_10": 0.8,
            "ndcg_at_5": 0.7,
            "max_regression_vs_best_baseline": 0.05,
            "max_outlet_concentration_at_10": 0.6,
        },
        "queries": queries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lengths = [len(record["query"].split()) for record in queries]
    hard_negatives = sum(
        1 for record in queries
        for grade in record["relevance"].values() if grade == 0
    )
    print(f"\nWrote {len(queries)} queries to {args.output}")
    print(f"  generated: {len(generated)}  carried over: {len(human)}")
    print(f"  dropped non-discriminative: {dropped_broad}  "
          f"dropped no-relevant: {dropped_empty}")
    print(f"  mean query length: {sum(lengths) / len(lengths):.1f} words")
    print(f"  hard negatives: {hard_negatives}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

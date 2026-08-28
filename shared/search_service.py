
import logging
import os

from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sqlalchemy import (
    func,
    select
)

from shared.database import SessionLocal
from shared.models import Articles


logger = logging.getLogger(__name__)

COLLECTION = "articles"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

PREFETCH_LIMIT = 50
RESULT_LIMIT = 15

RRF_K = 60
MAX_SEMANTIC_DISTANCE = 0.6

# Qdrant reports cosine similarity where pgvector reports cosine distance, so
# the distance ceiling above becomes a similarity floor here.
MIN_SEMANTIC_SIMILARITY = 1 - MAX_SEMANTIC_DISTANCE


class SearchService:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model
        self._qdrant = None
        self._sparse_model = None

    @property
    def qdrant(self) -> QdrantClient:
        """
        Builds the Qdrant client on first use. The cron only ever calls embed(),
        so connecting eagerly would let missing Qdrant credentials take down the
        scraper.

        :return: the configured Qdrant client
        """

        if self._qdrant is None:
            self._qdrant = QdrantClient(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
            )

        return self._qdrant

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        """
        Loads the BM25 model on first use. It must be the same model the
        backfill indexed with, or query terms hash to different ids than the
        stored documents.

        :return: the BM25 sparse embedding model
        """

        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(SPARSE_MODEL)

        return self._sparse_model

    @staticmethod
    def _deduplicate(
            articles: list[tuple[Articles, float]]
    ) -> list[tuple[Articles, float]]:
        """
        Removes duplicate articles while preserving their original ranking.
        The first occurrence of each article link is retained.

        :param articles: the ordered articles and their relevance scores
        :return: the ordered articles with duplicate links removed
        """

        deduplicated_articles = {}
        for article, score in articles:
            if article.link in deduplicated_articles:
                continue
            deduplicated_articles[article.link] = (article, score)

        return list(deduplicated_articles.values())

    @staticmethod
    def _hydrate(
            points: list[models.ScoredPoint]
    ) -> list[tuple[Articles, float]]:
        """
        Loads the articles behind a list of scored Qdrant points. Point ids
        mirror the Postgres primary key, so Qdrant only has to rank ids while
        the article bodies stay in one place.

        Postgres returns an IN query in whatever order it likes, so the rows are
        reordered to match Qdrant's ranking before being returned.

        :param points: the scored points to load articles for
        :return: the ranked articles and their Qdrant scores
        """

        if not points:
            return []

        point_ids = [point.id for point in points]

        with SessionLocal() as session:
            statement = select(Articles).where(Articles.id.in_(point_ids))
            articles = session.execute(statement).scalars().all()

        articles_by_id = {article.id: article for article in articles}

        ranked = [
            (articles_by_id[point.id], point.score)
            for point in points
            if point.id in articles_by_id
        ]

        return SearchService._deduplicate(ranked)

    def sparse_embed(self, text: str) -> models.SparseVector:
        """
        Builds the BM25 sparse vector for a query. Queries use query_embed
        rather than the backfill's passage_embed: passages carry term frequency
        weights, queries carry flat weights, and Qdrant supplies the inverse
        document frequency term itself through the collection's IDF modifier.

        :param text: the query to build a sparse vector for
        :return: the sparse vector of query term weights
        """

        embedding = next(iter(self.sparse_model.query_embed([text])))

        return models.SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

    def embed(self, text: str) -> list[float]:
        """
        Embeds an article text into an embedding vector.
        The text will be in this format:
        <headline>:<summary>

        :param text: the text to embed
        :return: a list of floats to represent embedding vector
        """

        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )

        return response.data[0].embedding

    def lexical_search(self, text: str) -> list[tuple[Articles, float]]:
        """
        Searches for articles that contain terms from the provided text.
        Results are ordered from most to least lexically relevant using
        cover-density ranking.

        :param text: the search query to compare against article text
        :return: up to 50 matching articles and their lexical relevance scores
        """

        article_text = func.concat(
            Articles.translated_headline,
            " ",
            Articles.summary
        )

        search_vector = func.to_tsvector("english", article_text)
        search_query = func.websearch_to_tsquery("english", text)
        search_result = None

        with SessionLocal() as session:
            score = func.ts_rank_cd(search_vector, search_query)
            statement = (
                select(Articles, score.label("rank"))
                .where(search_vector.op("@@")(search_query))
                .order_by(score.desc())
                .limit(50)
            )
            search_result = session.execute(statement).all()

        return self._deduplicate(search_result)


    def semantic_search(self, text: str) -> list[tuple[Articles, float]]:
        """
        Searches for articles that are semantically similar to the provided
        text. The query is embedded and compared against stored article
        embeddings using cosine distance. Results are ordered from most to
        least similar.

        :param text: the query to compare against article embeddings
        :return: the 50 closest articles and their cosine distances
        """

        query_vector = self.embed(text)
        distance = Articles.embedding.cosine_distance(query_vector)
        search_result = None

        with SessionLocal() as session:
            statement = (
                select(Articles, distance.label("distance"))
                .where(
                    Articles.embedding.is_not(None),
                    distance < MAX_SEMANTIC_DISTANCE
                )
                .order_by(distance)
                .limit(50)
            )

            search_result = session.execute(statement).all()

        return self._deduplicate(search_result)

    def hybrid_search(self, text: str) -> list[tuple[int, Articles]]:
        """
        Combines lexical and semantic search results using reciprocal rank
        fusion. Both arms are prefetched and fused inside Qdrant in a single
        round trip, so results appearing highly in either search receive a
        combined score and are ordered from most to least relevant.

        Qdrant fuses with k=1 rather than the k=60 used by the Postgres
        fallback, which weighs a top placement in one arm more heavily than
        appearing in both.

        Falls back to searching Postgres directly when Qdrant cannot be
        reached, since this path serves live user search.

        :param text: the query to search for across stored articles
        :return: the ranked result positions and their corresponding articles
        """

        try:
            search_result = self.qdrant.query_points(
                COLLECTION,
                prefetch=[
                    models.Prefetch(
                        query=self.embed(text),
                        using=DENSE_VECTOR,
                        limit=PREFETCH_LIMIT,
                        score_threshold=MIN_SEMANTIC_SIMILARITY,
                    ),
                    models.Prefetch(
                        query=self.sparse_embed(text),
                        using=SPARSE_VECTOR,
                        limit=PREFETCH_LIMIT,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=RESULT_LIMIT,
            ).points

        except Exception:
            logger.exception("Qdrant search failed, falling back to Postgres")
            return self.hybrid_search_backup(text)

        return [
            (rank, article)
            for rank, (article, _) in enumerate(
                self._hydrate(search_result), start=1
            )
        ]

    def hybrid_search_backup(self, text: str) -> list[tuple[int, Articles]]:
        """
        Fuses the Postgres lexical and semantic arms in Python using reciprocal
        rank fusion. Only used when Qdrant is unreachable.

        :param text: the query to search for across stored articles
        :return: the ranked result positions and their corresponding articles
        """

        lexical_search = self.lexical_search(text)
        semantic_search = self.semantic_search(text)

        scores = {}

        for i, (article, score) in enumerate(lexical_search):
            if article.link in scores:
                continue
            scores[article.link] = {}
            scores[article.link]["score"] = (1 / (RRF_K + (i + 1)))
            scores[article.link]["article"] = article

        for i, (article, score) in enumerate(semantic_search):
            if article.link in scores:
                scores[article.link]["score"] += (1 / (RRF_K + (i + 1)))
            else:
                scores[article.link] = {}
                scores[article.link]["score"] = (1 / (RRF_K + (i + 1)))
                scores[article.link]["article"] = article

        sorted_scores = dict(
            sorted(
                scores.items(),
                key=lambda item: item[1]["score"],
                reverse=True
            )
        )

        search_result = [
            (i + 1, article_object["article"]) for i,
            article_object in enumerate(sorted_scores.values())
        ]
        return search_result[:RESULT_LIMIT]
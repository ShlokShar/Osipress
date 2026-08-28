
import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sqlalchemy import select

from shared.database import SessionLocal
from shared.models import Articles
from shared.search_service import (
    COLLECTION,
    SPARSE_MODEL,
    SearchService,
)


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

EMBEDDING_SIZE = 1536
BATCH_SIZE = 128


def get_client() -> QdrantClient:
    """
    Builds a Qdrant client from the credentials stored in the environment.

    :return: the configured Qdrant client
    """

    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def create_collection(client: QdrantClient, reset: bool = False) -> None:
    """
    Creates the articles collection with a named dense vector and a named
    sparse vector.

    :param client: the Qdrant client to create the collection with
    :param reset: whether to drop an existing collection before creating
    """

    if reset:
        client.delete_collection(COLLECTION)
    elif client.collection_exists(COLLECTION):
        return

    client.create_collection(
        COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBEDDING_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )


def build_points(
        articles: list[Articles],
        search_service: SearchService,
        sparse_model: SparseTextEmbedding,
) -> list[models.PointStruct]:
    """
    Turns a batch of articles into Qdrant points. The point id mirrors the
    Postgres primary key.

    :param articles: the articles to convert into points
    :param search_service: the service used to embed articles missing a vector
    :param sparse_model: the BM25 model used to build sparse vectors
    :return: the points ready to be upserted
    """

    texts = [article.embedding_text() for article in articles]
    sparse_vectors = list(sparse_model.passage_embed(texts))

    points = []
    for article, text, sparse in zip(articles, texts, sparse_vectors):

        # reuse the stored embedding so Qdrant and Postgres never disagree
        dense = article.embedding
        if dense is None:
            dense = search_service.embed(text)

        points.append(
            models.PointStruct(
                id=article.id,
                vector={
                    "dense": list(dense),
                    "sparse": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload={
                    "link": article.link,
                    "source_id": article.source_id,
                    "captured_at": article.captured_at.isoformat(),
                    "tags": article.tags,
                },
            )
        )

    return points


def upsert_article(article: Articles, search_service: SearchService) -> None:
    """
    Writes a single already-saved article into Qdrant. The article is wrapped in
    a list so that this shares build_points with the full backfill, which keeps
    the two paths from drifting on payload shape, vector names, or which text
    gets indexed.

    :param article: the saved article, carrying the id Postgres assigned
    :param search_service: the service holding the Qdrant client and BM25 model
    """

    points = build_points(
        [article],
        search_service,
        search_service.sparse_model,
    )

    search_service.qdrant.upsert(COLLECTION, points=points, wait=True)


def backfill_vectors(reset: bool = False) -> int:
    """
    Loads every article out of Postgres and upserts it into Qdrant as a point
    carrying both a dense and a sparse vector. Upserts are keyed on the article
    id, so rerunning this repairs the collection rather than duplicating it.

    :param reset: whether to drop and recreate the collection first
    :return: the number of articles written
    """

    client = get_client()
    create_collection(client, reset=reset)

    search_service = SearchService()
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)

    written = 0
    with SessionLocal() as session:
        statement = select(Articles).order_by(Articles.id)
        articles = session.execute(statement).scalars().all()

        for start in range(0, len(articles), BATCH_SIZE):
            batch = articles[start:start + BATCH_SIZE]
            points = build_points(batch, search_service, sparse_model)
            client.upsert(COLLECTION, points=points, wait=True)
            written += len(points)
            print(f"upserted {written}/{len(articles)}")

    return written


if __name__ == "__main__":
    count = backfill_vectors()
    print(f"backfilled {count} articles into '{COLLECTION}'")

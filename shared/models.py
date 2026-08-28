
from collections import defaultdict
from datetime import (
    date,
    datetime,
    timedelta,
    timezone
)
from typing import (
    Any,
    Optional
)

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    DateTime,
    ForeignKey,
    func,
    String
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column
)

from shared.database import (
    Base,
    SessionLocal
)


class Countries(Base):
    __tablename__ = 'countries'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()

    @classmethod
    def get_country(
        cls,
        name: str = "",
        country_id: int = -1
    ) -> Optional["Countries"]:
        """
        Receives a country object from either the name or its database id.

        :param name: the country name
        :param country_id: the country id as per Postgres
        :return: the country object
        """

        with SessionLocal() as session:
            if country_id > 0:
                return session.query(cls).filter(cls.id == country_id).first()
            return session.query(cls).filter(cls.name == name).first()


class Sources(Base):
    __tablename__ = 'sources'

    id: Mapped[int] = mapped_column(primary_key=True)
    country_id: Mapped[int] = mapped_column(ForeignKey('countries.id'))
    name: Mapped[str] = mapped_column()
    link: Mapped[str] = mapped_column()
    political_leaning: Mapped[str] = mapped_column()

    @classmethod
    def get_source(cls, country_id: int, source_id: int) -> Mapped[int]:
        """
        Receives a source object from either the name or its database id.

        :param country_id: the country id as per Postgres
        :param source_id: the source id as per Postgres
        :return: the source object
        """

        with SessionLocal() as session:
            if source_id:
                return session.query(cls).filter(cls.id == source_id).first()
            return session.query(cls).filter(
                cls.country_id == country_id
            ).first()

    @classmethod
    def get_source_by_name(cls, country_id: int, name: str):
        """
        Receives a source object from either the name or its database id.

        :param country_id: the country id as per Postgres
        :param name: the source's name
        :return: the source object
        """

        with SessionLocal() as session:
            return session.query(cls).filter(
                cls.country_id == country_id, cls.name == name
            ).first()


class Articles(Base):
    __tablename__ = 'articles'

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey('sources.id'))
    headline: Mapped[str] = mapped_column()
    translated_headline: Mapped[str] = mapped_column()
    link: Mapped[str] = mapped_column()
    summary: Mapped[str] = mapped_column()
    references_original: Mapped[list[str]] = mapped_column(ARRAY(String))
    references_translated: Mapped[list[str]] = mapped_column(ARRAY(String))
    tags: Mapped[list[str]] = mapped_column(ARRAY(String))
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1536),
        nullable=True
    )
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )


    @staticmethod
    def add_article(article):
        """
        adds an article to the Postgres database.

        The commit expires every attribute, so the article is refreshed while
        the session is still open. Callers need this to read the id Postgres
        assigned on insert, which is what keys the article's point in Qdrant.

        :param article: the article object (includes the foreign source key)
        :return: the article object
        """

        with SessionLocal() as session:
            session.add(article)
            session.commit()
            session.refresh(article)

        return article

    def embedding_text(self) -> str:
        """
        Retrieves the translated headline and article summary to be embedded.
        :return: the text to be embedded
        """

        return f"{self.translated_headline}:{self.summary}"

    def to_dict(self):
        """
        Returns a dictionary version of the data.

        :return: a dictionary version of the data
        """

        return {
            "id": self.id,
            "source_id": self.source_id,
            "headline": self.headline,
            "translated_headline": self.translated_headline,
            "link": self.link,
            "summary": self.summary,
            "references_original": self.references_original,
            "references_translated": self.references_translated,
            "tags": self.tags,
        }


def get_headlines_by_country(
    target_date: Optional[date] = None
) -> defaultdict[Any, dict[Any, Any]]:
    """
    Returns a dictionary version of the data based on the date the articles
    were captured.

    :param target_date: the target date to filter on
    :return: a list of the original headlines in English
    """

    if target_date is None:
        target_date = datetime.now(timezone.utc).date()

    start = datetime.combine(
        target_date,
        datetime.min.time(),
        tzinfo=timezone.utc
    )
    end = start + timedelta(days=1)

    with SessionLocal() as session:
        latest_per_source = (
            session.query(
                Articles.source_id,
                func.max(Articles.captured_at).label("latest_captured_at")
            )
            .filter(Articles.captured_at >= start, Articles.captured_at < end)
            .group_by(Articles.source_id)
            .subquery()
        )

        results = (
            session.query(
                Countries.name,
                Sources.name,
                Sources.political_leaning,
                Articles
            )
            .join(Sources, Sources.country_id == Countries.id)
            .join(Articles, Articles.source_id == Sources.id)
            .join(
                latest_per_source,
                (Articles.source_id == latest_per_source.c.source_id)
                & (
                    Articles.captured_at ==
                    latest_per_source.c.latest_captured_at
                )
            )
            .all()
        )

        output = defaultdict(dict)
        for country_name, source_name, political_leaning, article in results:
            if source_name not in output[country_name]:
                output[country_name][source_name] = {
                    "political_leaning": political_leaning,
                    "articles": [],
                }
            output[country_name][source_name]["articles"].append(
                article.to_dict()
            )

    return output


def get_sources_by_ids(source_ids: list[int]) -> dict[int, dict[str, str]]:
    """
    Returns the outlet and country details for the given source ids. Search
    results carry only a source id, so this resolves the outlet name, its
    country and its political leaning in a single query.

    :param source_ids: the source ids as per Postgres
    :return: a mapping of source id to its outlet, country and leaning
    """

    if not source_ids:
        return {}

    with SessionLocal() as session:
        results = (
            session.query(
                Sources.id,
                Sources.name,
                Sources.political_leaning,
                Countries.name
            )
            .join(Countries, Sources.country_id == Countries.id)
            .filter(Sources.id.in_(set(source_ids)))
            .all()
        )

    return {
        source_id: {
            "outlet": source_name,
            "political_leaning": political_leaning,
            "country": country_name,
        }
        for source_id, source_name, political_leaning, country_name in results
    }

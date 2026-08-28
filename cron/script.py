
import json
from datetime import (
    datetime,
    timezone
)
from pathlib import Path
from typing import (
    Optional,
    TypedDict,
)

import feedparser
from feedparser.util import FeedParserDict

from cron.ai.service import AIService
from cron.util.article import (
    get_article_content,
    is_safe_article_url
)
from cron.util.backfill_vectors import upsert_article
from cron.util.log import add_log
from cron.util.translation import (
    translate_references
)
from shared.models import (
    Articles,
    Countries,
    Sources
)
from shared.search_service import SearchService


class SourceConfig(TypedDict, total=False):
    name: str
    url: str
    rss: bool


SourcesData = dict[str, dict[str, SourceConfig]]
SetupResult = tuple[AIService, SearchService, datetime, SourcesData]
SourceResult = tuple[Sources, str, str]
ArticleInfo = tuple[
    str,
    str,
    str,
    str,
    list[str],
    list[str],
    list[str],
    Optional[list[float]],
]


MAX_ARTICLES = 3
SOURCES_PATH = Path(__file__).resolve().parent / "sources.json"


def setup() -> SetupResult:
    """
    Initiates and declares the ai_service and vector_service. Opens the JSON
    file and saves it as data and sets the run time for the script.

    :return: the ai_service and vector_service instances, run_time and data
    variables
    """

    try:
        ai_service = AIService()
        vector_service = SearchService()
    except Exception as exception:
        add_log(
            f"AI or Vector Service failed to initialize "
            f"({type(exception).__name__}): {exception}"
        )
        raise

    run_time = datetime.now(timezone.utc)
    try:
        with SOURCES_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exception:
        add_log(
            f"Sources file failed to load "
            f"({type(exception).__name__}): {exception}"
        )
        raise

    return ai_service, vector_service, run_time, data


def get_country(country: str) -> Optional[Countries]:
    """
    Finds the country object in the database from the name in the JSON data.

    :param country: the country name as per the JSON data
    :return: the matching country database object, or None if not found
    """

    try:
        country_object = Countries.get_country(country)
    except Exception as exception:
        add_log(
            f"{country}: failed to load country from database "
            f"({type(exception).__name__}): {exception}"
        )
        return None

    if not country_object:
        add_log(f"{country}: country is missing from database")
        return None
    return country_object


def get_source(
    country_object: Countries,
    country: str,
    source: str,
    config: SourceConfig,
) -> Optional[SourceResult]:
    """
    Gets the source object from the database based on the country name.

    :param country_object: the country object from the database
    :param country: the country name as per the JSON data
    :param source: the source name as per the JSON data
    :param config: the SourceConfig instance
    :return: the source object, or None if not found
    """

    try:
        source_name = config["name"]
        url = config["url"]
    except (KeyError, TypeError) as exception:
        add_log(
            f"{country} / {source}: invalid source configuration "
            f"({type(exception).__name__}): {exception}"
        )
        return None

    try:
        source_object = Sources.get_source_by_name(
            country_object.id,
            source_name
        )
    except Exception as exception:
        add_log(
            f"{country} / {source_name}: failed to load source from "
            f"database ({type(exception).__name__}): {exception}"
        )
        return None

    if not source_object:
        add_log(
            f"{country} / {source_name}: source is missing from database"
        )
        return None

    return source_object, source_name, url


def validate_feed(
    url: str,
    country: str,
    source_name: str,
) -> Optional[FeedParserDict]:
    """
    Gets the feed information and checks if the request / response was valid.

    :param url: the RSS feed URL
    :param country: the country name as per the JSON data
    :param source_name: the source name as per the JSON data
    :return: the feed object, or None if not found
    """

    try:
        feed = feedparser.parse(url)
    except Exception as exception:
        add_log(
            f"{country} / {source_name}: feed failed to load "
            f"({type(exception).__name__}): {exception}"
        )
        return None

    feed_status = getattr(feed, "status", None)
    if feed_status != 200:
        add_log(
            f"{country} / {source_name}: feed returned status "
            f"{feed_status}"
        )
        return None

    if not feed.entries:
        add_log(f"{country} / {source_name}: feed returned no articles")
        return None
    return feed


def get_article_info(
    entry: FeedParserDict,
    ai_service: AIService,
    vector_service: SearchService,
) -> Optional[ArticleInfo]:
    """
    Retrieves all the information needed to create an article instance,
    including the article vector.

    :param entry: the RSS feed entry to process
    :param ai_service: the service used to classify and process article text
    :param vector_service: the service used to embed the article overview
    :return: the processed article information, or None if processing fails
    """

    headline = "Unknown headline"
    try:
        headline = entry.title
        relevant = ai_service.classify(headline)
        link = entry.link

        if not is_safe_article_url(link):
            return None
        if not relevant:
            return None

        article_content = get_article_content(link)
        if article_content == "empty article.":
            return None

        translated_headline = ai_service.translate_headline(headline)
        processed_article = ai_service.summarize(
            headline,
            article_content,
        )

        if not processed_article:
            return None

        summary = processed_article.summary
        references_original = processed_article.references
        references = processed_article.references_for_translation
        tags = processed_article.tags

        references_translated = translate_references(
            references
        )

        article_overview = (
            f"{translated_headline}:{summary}"
        )

        try:
            vector = vector_service.embed(article_overview)
        except Exception as exception:
            add_log(
                f"exception: {type(exception).__name__}: "
                f"{exception}; failed to embed article overview: "
                f"{article_overview}"
            )
            vector = None

        return (
            headline,
            translated_headline,
            link,
            summary,
            references_original,
            references_translated,
            tags,
            vector
        )
    except Exception as exception:
        add_log(
            f"{headline}: failed to process article "
            f"({type(exception).__name__}): {exception}"
        )
        return None


def main() -> None:
    ai_service, vector_service, run_time, data = setup()

    for country, sources in data.items():
        country_object = get_country(country)
        if not country_object:
            continue

        for source, config in sources.items():
            source_result = get_source(
                country_object,
                country,
                source,
                config,
            )
            if not source_result:
                continue

            source_object, source_name, url = source_result
            feed = validate_feed(url, country, source_name)
            if not feed:
                continue

            saved_articles = 0

            for entry in feed.entries:
                if saved_articles >= MAX_ARTICLES:
                    break

                article_info = get_article_info(
                    entry,
                    ai_service,
                    vector_service
                )
                if not article_info:
                    continue

                (
                    headline,
                    translated_headline,
                    link,
                    summary,
                    references,
                    translated,
                    tags,
                    vector
                ) = article_info

                article = Articles(
                    source_id=source_object.id,
                    headline=headline,
                    translated_headline=translated_headline,
                    link=link,
                    summary=summary,
                    references_original=references,
                    references_translated=translated,
                    tags=tags,
                    embedding=vector,
                    captured_at=run_time,
                )

                try:
                    article = Articles.add_article(article)
                except Exception as exception:
                    add_log(
                        f"{country} / {source_name} / {headline}: save failed "
                        f"({type(exception).__name__}): {exception}"
                    )
                    continue

                saved_articles += 1

                # Postgres is the source of truth, so a failed upsert is logged
                # and the article stays saved. It is invisible to search until
                # the next backfill_vectors() run repairs it.
                try:
                    upsert_article(article, vector_service)
                except Exception as exception:
                    add_log(
                        f"{country} / {source_name} / {headline}: Qdrant "
                        f"upsert failed ({type(exception).__name__}): "
                        f"{exception}"
                    )

            if saved_articles < MAX_ARTICLES:
                add_log(
                    f"{country} / {source_name}: saved {saved_articles}/"
                    f"{MAX_ARTICLES} articles"
                )


if __name__ == "__main__":
    main()

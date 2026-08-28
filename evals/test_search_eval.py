import math
import unittest
from types import SimpleNamespace

from shared.search_service import RRF_K, SearchService
from evals.search_eval import (
    aggregate_by_category,
    average_precision,
    compare_methods,
    ndcg,
    query_metrics,
    reciprocal_rank,
    unique_links,
)


class SearchMetricTests(unittest.TestCase):
    def setUp(self):
        self.relevance = {"a": 3, "b": 1}

    def test_perfect_ranking(self):
        metrics = query_metrics(["a", "b", "c"], self.relevance)
        self.assertEqual(metrics["mrr_at_10"], 1.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["map_at_3"], 1.0)
        self.assertEqual(metrics["ndcg_at_3"], 1.0)

    def test_missing_results_score_zero(self):
        self.assertEqual(reciprocal_rank(["x"], self.relevance, 10), 0.0)
        self.assertEqual(average_precision(["x"], self.relevance, 10), 0.0)
        self.assertEqual(ndcg(["x"], self.relevance, 10), 0.0)

    def test_ndcg_rewards_grade_order(self):
        ideal = ndcg(["a", "b"], self.relevance, 10)
        reversed_score = ndcg(["b", "a"], self.relevance, 10)
        self.assertEqual(ideal, 1.0)
        self.assertLess(reversed_score, ideal)

    def test_cutoff_is_respected(self):
        self.assertEqual(reciprocal_rank(["x", "a"], self.relevance, 1), 0.0)
        self.assertEqual(reciprocal_rank(["x", "a"], self.relevance, 2), 0.5)

    def test_unique_links_normalizes_all_search_shapes(self):
        a = SimpleNamespace(link="a")
        b = SimpleNamespace(link="b")
        self.assertEqual(unique_links([(a, 0.9), (a, 0.8), (b, 0.7)]),
                         ["a", "b"])
        self.assertEqual(unique_links([(1, a), (2, b)]), ["a", "b"])

    def test_category_aggregation_and_method_comparison(self):
        result = {
            "category": "paraphrase",
            "error": None,
            "metrics": {"mrr_at_10": 1.0},
            "latency_ms": 10.0,
        }
        categories = aggregate_by_category([result])
        self.assertEqual(
            categories["paraphrase"]["metrics"]["mrr_at_10"], 1.0
        )

        methods = {
            "lexical": {"aggregate": {"metrics": {
                "mrr_at_10": 0.5,
                "map_at_10": 0.5,
                "ndcg_at_10": 0.5,
                "recall_at_10": 0.5,
            }}},
            "semantic": {"aggregate": {"metrics": {
                "mrr_at_10": 0.8,
                "map_at_10": 0.8,
                "ndcg_at_10": 0.8,
                "recall_at_10": 0.8,
            }}},
            "hybrid": {"aggregate": {"metrics": {
                "mrr_at_10": 0.9,
                "map_at_10": 0.9,
                "ndcg_at_10": 0.9,
                "recall_at_10": 0.9,
            }}},
        }
        comparison = compare_methods(methods)
        self.assertEqual(
            comparison["hybrid_minus_semantic"]["mrr_at_10"], 0.1
        )


class HybridSearchTests(unittest.TestCase):
    def test_rrf_rewards_results_present_in_both_rankings(self):
        lexical_only = SimpleNamespace(link="lexical")
        shared = SimpleNamespace(link="shared")
        semantic_only = SimpleNamespace(link="semantic")

        service = SearchService.__new__(SearchService)
        service.lexical_search = lambda _: [
            (lexical_only, 10.0),
            (shared, 9.0),
        ]
        service.semantic_search = lambda _: [
            (semantic_only, 0.1),
            (shared, 0.2),
        ]

        results = service.hybrid_search_backup("query")

        self.assertEqual(results[0], (1, shared))
        expected_score = 2 / (RRF_K + 2)
        competing_score = 1 / (RRF_K + 1)
        self.assertGreater(expected_score, competing_score)
        self.assertEqual([rank for rank, _ in results], [1, 2, 3])

    def test_hybrid_limits_results_to_fifteen(self):
        articles = [SimpleNamespace(link=str(index)) for index in range(30)]
        service = SearchService.__new__(SearchService)
        service.lexical_search = lambda _: [
            (article, float(index))
            for index, article in enumerate(articles)
        ]
        service.semantic_search = lambda _: []

        results = service.hybrid_search_backup("query")

        self.assertEqual(len(results), 15)
        self.assertTrue(all(
            math.isfinite(rank) for rank, _ in results
        ))


if __name__ == "__main__":
    unittest.main()

"""Report and compare search eval runs.

With one report, prints every metric the run recorded. With two, pairs them
query by query so a backend change can be judged properly.

WHY PAIRING MATTERS
-------------------
Comparing two aggregate numbers ("nDCG went from 0.906 to 0.921") cannot tell
you whether a backend is better. On ~126 queries a swing of a few points is
routinely noise, and an aggregate hides the case that matters most: a change
that lifts most queries slightly while destroying a handful. Because both runs
score the SAME queries, the differences are paired, which supports a far
stronger test than comparing two independent means.

So this reports, per metric:
  * the aggregate delta
  * how many queries improved, regressed, or tied
  * a 95% confidence interval on the mean delta, by paired bootstrap
  * the biggest individual regressions, which are what you actually debug

A paired bootstrap resamples QUERIES (not scores) with replacement 10,000
times and recomputes the mean delta each time. If the resulting interval
straddles zero, the change is indistinguishable from noise no matter how good
the headline number looks.

Usage:
    PYTHONPATH=. .venv/bin/python evals/compare_search.py results/a.json
    PYTHONPATH=. .venv/bin/python evals/compare_search.py results/a.json \
        results/b.json --method hybrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy


# Ordered so the printed table reads from "did we find it at all" through to
# "how well was the whole list ranked".
METRIC_FAMILIES = (
    ("Binary retrieval", ("hit_rate", "precision", "recall")),
    ("Ranking quality", ("mrr", "map", "ndcg")),
)
CUTOFFS = (1, 3, 5, 10, 15)

DIVERSITY_METRICS = (
    ("distinct_outlets_at_10", "distinct outlets in top 10"),
    ("distinct_countries_at_10", "distinct countries in top 10"),
    ("outlet_concentration_at_10", "share held by top outlet (lower better)"),
)

# Metrics where a LOWER number is better, so deltas must be read inverted.
LOWER_IS_BETTER = {
    "outlet_concentration_at_10",
    *(f"false_positive_rate_at_{cutoff}" for cutoff in CUTOFFS),
}

BOOTSTRAP_SAMPLES = 10000


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def print_run(report: dict[str, Any]) -> None:
    """Prints every recorded metric for a single run."""

    golden = report["golden_set"]
    mix = report.get("query_mix", {})
    coverage = report.get("corpus_coverage", {})

    print("=" * 72)
    print(f"RUN  {report['evaluated_at']}")
    print(f"  golden set   {golden['file']} v{golden['version']} "
          f"({golden['query_count']} queries)")
    print(f"  embedding    {report['search_config'].get('embedding_model')}")
    print(f"  corpus       {coverage.get('corpus_unique_links')} unique links, "
          f"{coverage.get('corpus_embedded_unique_links')} embedded")
    if mix:
        print(f"  query mix    {mix.get('by_length')} "
              f"mean {mix.get('mean_query_words')} words")
        print(f"  negatives    {mix.get('queries_with_hard_negatives')} queries, "
              f"{mix.get('hard_negative_judgments')} judgments")
    print("=" * 72)

    methods = list(report["methods"])

    for title, families in METRIC_FAMILIES:
        print(f"\n{title}")
        header = "  " + "metric".ljust(22) + "".join(
            name.rjust(11) for name in methods
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for family in families:
            for cutoff in CUTOFFS:
                key = f"{family}_at_{cutoff}"
                cells = "".join(
                    f"{report['methods'][name]['aggregate']['metrics'].get(key, float('nan')):11.4f}"
                    for name in methods
                )
                print("  " + key.ljust(22) + cells)

    print("\nResult diversity and hard negatives")
    print("  " + "metric".ljust(38) + "".join(
        name.rjust(11) for name in methods))
    for key, label in DIVERSITY_METRICS:
        cells = "".join(
            f"{report['methods'][name]['aggregate']['metrics'].get(key, float('nan')):11.4f}"
            for name in methods
        )
        print("  " + label.ljust(38)[:38] + cells)
    for cutoff in (5, 10):
        key = f"false_positive_rate_at_{cutoff}"
        if any(key in report["methods"][name]["aggregate"]["metrics"]
               for name in methods):
            cells = "".join(
                f"{report['methods'][name]['aggregate']['metrics'].get(key, float('nan')):11.4f}"
                for name in methods
            )
            print("  " + f"hard-negative leak@{cutoff} (lower better)".ljust(38)
                  + cells)

    print("\nLatency (ms)")
    for stat in ("mean", "p50", "p95", "max"):
        cells = "".join(
            f"{report['methods'][name]['aggregate']['latency_ms'].get(stat, 0):11.2f}"
            for name in methods
        )
        print("  " + stat.ljust(22) + cells)

    for axis, label in (("by_length", "query length"),
                        ("by_category", "category")):
        if axis not in report["methods"][methods[0]]:
            continue
        print(f"\nnDCG@10 by {label}")
        buckets = sorted(report["methods"][methods[0]][axis])
        width = max(len(bucket) for bucket in buckets) + 2
        print("  " + "bucket".ljust(width) + "n".rjust(5)
              + "".join(name.rjust(11) for name in methods))
        for bucket in buckets:
            count = report["methods"][methods[0]][axis][bucket][
                "successful_queries"]
            cells = "".join(
                f"{report['methods'][name][axis][bucket]['metrics'].get('ndcg_at_10', float('nan')):11.4f}"
                for name in methods
            )
            print("  " + bucket.ljust(width) + str(count).rjust(5) + cells)

    gate = report.get("quality_gate", {})
    print(f"\nQuality gate: {'PASS' if gate.get('passed') else 'FAIL'}")
    for failure in gate.get("failures", []):
        print(f"  - {failure}")


def paired_bootstrap(deltas: numpy.ndarray,
                     samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float, float]:
    """
    Bootstraps a 95% confidence interval on the mean paired delta.

    Resamples the QUERIES with replacement, so the interval reflects how much
    the result depends on which queries happen to be in the golden set. An
    interval containing zero means the change is not distinguishable from
    noise on this set.

    :param deltas: the per-query differences between two runs
    :param samples: how many bootstrap resamples to draw
    :return: the mean delta and the lower and upper bounds of the interval
    """

    if len(deltas) == 0:
        return 0.0, 0.0, 0.0

    generator = numpy.random.default_rng(20260825)
    indices = generator.integers(0, len(deltas), size=(samples, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return (
        float(deltas.mean()),
        float(numpy.percentile(means, 2.5)),
        float(numpy.percentile(means, 97.5)),
    )


def compare(baseline: dict[str, Any], candidate: dict[str, Any],
            method: str, top_regressions: int) -> None:
    """Pairs two runs query by query for one retrieval method."""

    base_queries = {
        record["id"]: record
        for record in baseline["methods"][method]["queries"]
    }
    new_queries = {
        record["id"]: record
        for record in candidate["methods"][method]["queries"]
    }
    shared = sorted(set(base_queries) & set(new_queries))

    print("=" * 72)
    print(f"PAIRED COMPARISON  method={method}")
    print(f"  baseline   {baseline['evaluated_at']} "
          f"({baseline['golden_set']['file']} "
          f"v{baseline['golden_set']['version']})")
    print(f"  candidate  {candidate['evaluated_at']} "
          f"({candidate['golden_set']['file']} "
          f"v{candidate['golden_set']['version']})")
    print(f"  {len(shared)} shared queries")

    if baseline["golden_set"].get("sha256") != candidate["golden_set"].get("sha256"):
        print("  WARNING: golden sets differ. Absolute metrics are NOT "
              "comparable across different judgment sets; only run this on "
              "two runs of the same golden set.")

    only_base = set(base_queries) - set(new_queries)
    only_new = set(new_queries) - set(base_queries)
    if only_base or only_new:
        print(f"  NOTE: {len(only_base)} queries only in baseline, "
              f"{len(only_new)} only in candidate; both excluded.")
    print("=" * 72)

    metric_keys = [
        f"{family}_at_{cutoff}"
        for _, families in METRIC_FAMILIES
        for family in families
        for cutoff in CUTOFFS
    ] + [key for key, _ in DIVERSITY_METRICS] + [
        f"false_positive_rate_at_{cutoff}" for cutoff in (5, 10)
    ]

    header = ("  " + "metric".ljust(24) + "baseline".rjust(10)
              + "candidate".rjust(11) + "delta".rjust(9)
              + "95% CI".rjust(18) + "  w/l/t")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key in metric_keys:
        pairs = [
            (base_queries[qid]["metrics"][key], new_queries[qid]["metrics"][key])
            for qid in shared
            if key in base_queries[qid].get("metrics", {})
            and key in new_queries[qid].get("metrics", {})
        ]
        if not pairs:
            continue

        base_values = numpy.array([pair[0] for pair in pairs], dtype=float)
        new_values = numpy.array([pair[1] for pair in pairs], dtype=float)
        deltas = new_values - base_values
        mean_delta, low, high = paired_bootstrap(deltas)

        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        ties = int((deltas == 0).sum())
        if key in LOWER_IS_BETTER:
            wins, losses = losses, wins

        significant = "" if low <= 0 <= high else " *"
        print("  " + key.ljust(24)
              + f"{base_values.mean():10.4f}"
              + f"{new_values.mean():11.4f}"
              + f"{mean_delta:+9.4f}"
              + f"[{low:+.4f},{high:+.4f}]".rjust(18)
              + f"  {wins}/{losses}/{ties}{significant}")

    print("\n  * = 95% CI excludes zero (change is distinguishable from noise)")
    if any(key in LOWER_IS_BETTER for key in metric_keys):
        print("  win/loss counts are inverted for lower-is-better metrics")

    print("\nLatency (ms)")
    for stat in ("mean", "p50", "p95", "max"):
        before = baseline["methods"][method]["aggregate"]["latency_ms"].get(stat, 0)
        after = candidate["methods"][method]["aggregate"]["latency_ms"].get(stat, 0)
        print(f"  {stat.ljust(24)}{before:10.2f}{after:11.2f}"
              f"{after - before:+9.2f}")

    # Individual regressions are what you debug; the aggregate hides them.
    regressions = sorted(
        (
            (new_queries[qid]["metrics"].get("ndcg_at_10", 0.0)
             - base_queries[qid]["metrics"].get("ndcg_at_10", 0.0), qid)
            for qid in shared
            if base_queries[qid].get("metrics") and new_queries[qid].get("metrics")
        )
    )
    print(f"\nWorst nDCG@10 regressions (top {top_regressions})")
    for delta, qid in regressions[:top_regressions]:
        if delta >= 0:
            print("  none")
            break
        print(f"  {delta:+.4f}  {qid}  {base_queries[qid]['query']!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--method", default="hybrid")
    parser.add_argument("--top-regressions", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load(args.baseline)

    if args.candidate is None:
        print_run(baseline)
        return 0

    compare(baseline, load(args.candidate), args.method, args.top_regressions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

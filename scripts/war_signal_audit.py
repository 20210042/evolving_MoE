#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "hard-WAR": "results/acc/seed20210111/acc/seed20210111",
    "latest": "results/acc/seed20211004/acc/seed20211004",
}

DEFAULT_WINDOWS = [1, 2, 4, 8, 16]


def load_rows(path: str) -> list[dict]:
    p = ROOT / path / "evolution_log.jsonl"
    return [json.loads(line) for line in p.open(encoding="utf-8")]


def longest_stable_size_run(rows: list[dict]) -> list[int]:
    sizes = [len(r["roster_after"]) for r in rows]
    common = Counter(sizes).most_common(1)[0][0]

    best = []
    cur = []

    for i, size in enumerate(sizes):
        if size == common:
            cur.append(i)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []

    if len(cur) > len(best):
        best = cur

    return best


def common_members(rows: list[dict], idx: list[int]) -> list[str]:
    members = set(rows[idx[0]]["war"])
    for i in idx[1:]:
        members &= set(rows[i]["war"])
    return sorted(members)


def mean_vector(
    rows: list[dict],
    batch_indices: list[int],
    members: list[str],
) -> list[float]:
    return [
        st.mean(rows[i]["war"][m] for i in batch_indices)
        for m in members
    ]


def pearson(x: list[float], y: list[float]) -> float:
    mx = st.mean(x)
    my = st.mean(y)

    dx = [v - mx for v in x]
    dy = [v - my for v in y]

    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))

    if sx <= 1e-12 or sy <= 1e-12:
        return math.nan

    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)

    p = 0
    while p < len(order):
        q = p + 1
        while q < len(order) and values[order[q]] == values[order[p]]:
            q += 1

        rank = (p + 1 + q) / 2.0

        for j in range(p, q):
            ranks[order[j]] = rank

        p = q

    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def pairwise_order_agreement(
    x: list[float],
    y: list[float],
) -> tuple[float, float]:
    total = 0
    comparable = 0
    agree = 0

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            total += 1

            dx = x[i] - x[j]
            dy = y[i] - y[j]

            if abs(dx) <= 1e-12 or abs(dy) <= 1e-12:
                continue

            comparable += 1

            if dx * dy > 0:
                agree += 1

    agreement = agree / comparable if comparable else math.nan
    coverage = comparable / total if total else math.nan

    return agreement, coverage


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)

    if not xs:
        return math.nan

    if len(xs) == 1:
        return xs[0]

    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return xs[lo]

    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def summarize(xs: list[float]) -> dict[str, float]:
    valid = [x for x in xs if not math.isnan(x)]

    if not valid:
        return {
            "mean": math.nan,
            "median": math.nan,
            "p05": math.nan,
            "p95": math.nan,
            "valid_rate": 0.0,
        }

    return {
        "mean": st.mean(valid),
        "median": st.median(valid),
        "p05": quantile(valid, 0.05),
        "p95": quantile(valid, 0.95),
        "valid_rate": len(valid) / len(xs),
    }


def monte_carlo_reproducibility(
    rows: list[dict],
    idx: list[int],
    members: list[str],
    window: int,
    repeats: int,
    seed: int,
) -> dict:
    if 2 * window > len(idx):
        return {}

    rng = random.Random(seed)

    pearsons = []
    spearmans = []
    agreements = []
    coverages = []

    for _ in range(repeats):
        chosen = rng.sample(idx, 2 * window)

        a_idx = chosen[:window]
        b_idx = chosen[window:]

        a = mean_vector(rows, a_idx, members)
        b = mean_vector(rows, b_idx, members)

        pearsons.append(pearson(a, b))
        spearmans.append(spearman(a, b))

        agreement, coverage = pairwise_order_agreement(a, b)
        agreements.append(agreement)
        coverages.append(coverage)

    return {
        "pearson": summarize(pearsons),
        "spearman": summarize(spearmans),
        "order_agreement": summarize(agreements),
        "order_coverage": summarize(coverages),
        "raw": {
            "pearson": pearsons,
            "spearman": spearmans,
            "order_agreement": agreements,
            "order_coverage": coverages,
        },
    }


def contiguous_nonoverlap(
    rows: list[dict],
    idx: list[int],
    members: list[str],
    window: int,
) -> dict:
    n_windows = len(idx) // window

    vectors = []

    for w in range(n_windows):
        batch_indices = idx[w * window:(w + 1) * window]
        vectors.append(mean_vector(rows, batch_indices, members))

    pearsons = []
    spearmans = []
    agreements = []
    coverages = []

    for a, b in zip(vectors, vectors[1:]):
        pearsons.append(pearson(a, b))
        spearmans.append(spearman(a, b))

        agreement, coverage = pairwise_order_agreement(a, b)
        agreements.append(agreement)
        coverages.append(coverage)

    return {
        "n_windows": len(vectors),
        "n_pairs": max(0, len(vectors) - 1),
        "pearson": summarize(pearsons),
        "spearman": summarize(spearmans),
        "order_agreement": summarize(agreements),
        "order_coverage": summarize(coverages),
    }


def fmt(x: float) -> str:
    if math.isnan(x):
        return "NA"
    return f"{x:+.3f}"


def pct(x: float) -> str:
    if math.isnan(x):
        return "NA"
    return f"{100 * x:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
    )
    args = parser.parse_args()

    out_md = ROOT / "results/acc/war_signal_repro_compare.md"
    out_csv = ROOT / "results/acc/war_signal_repro_trials.csv"

    run_data = {}

    for name, path in RUNS.items():
        rows = load_rows(path)
        idx = longest_stable_size_run(rows)

        if len(idx) < 2:
            raise RuntimeError(f"{name}: stable segment too short")

        members = common_members(rows, idx)

        if len(members) < 3:
            raise RuntimeError(f"{name}: too few common WAR members")

        run_data[name] = {
            "rows": rows,
            "idx": idx,
            "members": members,
            "roster_size": len(rows[idx[0]]["roster_after"]),
        }

    lines = []

    lines.append("# Hard-WAR vs Latest WAR — Independent-Sample Signal Reproducibility")
    lines.append("")
    lines.append(
        "각 trial에서 안정 로스터 구간의 서로 다른 2k개 batch를 비복원 추출하고, "
        "k개씩 두 독립 집합으로 나누어 agent별 평균 WAR vector를 계산한다."
    )
    lines.append(
        "두 집합은 batch를 공유하지 않는다. 따라서 overlapping moving-window의 "
        "기계적인 자기상관을 제거한 signal reproducibility 분석이다."
    )
    lines.append("")

    lines.append("## Run summary")
    lines.append("")
    lines.append("| Run | Stable steps | Roster size | Common WAR agents |")
    lines.append("|---|---:|---:|---:|")

    for name, d in run_data.items():
        lines.append(
            f"| {name} | {len(d['idx'])} | "
            f"{d['roster_size']} | {len(d['members'])} |"
        )

    lines.append("")
    lines.append("## Monte Carlo independent split")
    lines.append("")
    lines.append(
        f"각 k마다 {args.repeats:,}회 반복. 괄호는 trial 분포의 5–95 percentile."
    )
    lines.append("")
    lines.append(
        "| k batches / estimate | Run | Pearson r | Spearman rho | "
        "Pairwise order agreement | Order coverage |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|")

    all_results = {}

    for window in args.windows:
        for run_no, (name, d) in enumerate(run_data.items()):
            result = monte_carlo_reproducibility(
                rows=d["rows"],
                idx=d["idx"],
                members=d["members"],
                window=window,
                repeats=args.repeats,
                seed=args.seed + window * 100 + run_no,
            )

            if not result:
                continue

            all_results[(name, window)] = result

            p = result["pearson"]
            s = result["spearman"]
            a = result["order_agreement"]
            c = result["order_coverage"]

            lines.append(
                f"| {window} | {name} | "
                f"{fmt(p['mean'])} ({fmt(p['p05'])}, {fmt(p['p95'])}) | "
                f"{fmt(s['mean'])} ({fmt(s['p05'])}, {fmt(s['p95'])}) | "
                f"{pct(a['mean'])} | "
                f"{pct(c['mean'])} |"
            )

    lines.append("")
    lines.append("## Latest − Hard-WAR")
    lines.append("")
    lines.append(
        "| k | Delta Pearson | Delta Spearman | "
        "Delta order agreement | Delta coverage |"
    )
    lines.append("|---:|---:|---:|---:|---:|")

    for window in args.windows:
        hard = all_results.get(("hard-WAR", window))
        latest = all_results.get(("latest", window))

        if hard is None or latest is None:
            continue

        dp = (
            latest["pearson"]["mean"]
            - hard["pearson"]["mean"]
        )
        ds = (
            latest["spearman"]["mean"]
            - hard["spearman"]["mean"]
        )
        da = (
            latest["order_agreement"]["mean"]
            - hard["order_agreement"]["mean"]
        )
        dc = (
            latest["order_coverage"]["mean"]
            - hard["order_coverage"]["mean"]
        )

        lines.append(
            f"| {window} | {fmt(dp)} | {fmt(ds)} | "
            f"{100 * da:+.1f} pp | {100 * dc:+.1f} pp |"
        )

    lines.append("")
    lines.append("## Strict contiguous non-overlapping windows")
    lines.append("")
    lines.append(
        "Monte Carlo 결과의 sanity check로, 안정구간을 시간순으로 k-batch block으로 "
        "자른 뒤 서로 겹치지 않는 인접 block끼리도 비교한다."
    )
    lines.append("")
    lines.append(
        "| k | Run | windows | comparisons | Pearson | Spearman | "
        "Order agreement | Coverage |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")

    for window in args.windows:
        for name, d in run_data.items():
            result = contiguous_nonoverlap(
                rows=d["rows"],
                idx=d["idx"],
                members=d["members"],
                window=window,
            )

            lines.append(
                f"| {window} | {name} | "
                f"{result['n_windows']} | {result['n_pairs']} | "
                f"{fmt(result['pearson']['mean'])} | "
                f"{fmt(result['spearman']['mean'])} | "
                f"{pct(result['order_agreement']['mean'])} | "
                f"{pct(result['order_coverage']['mean'])} |"
            )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "window",
                "trial",
                "pearson",
                "spearman",
                "order_agreement",
                "order_coverage",
            ]
        )

        for (name, window), result in all_results.items():
            raw = result["raw"]

            for trial in range(args.repeats):
                writer.writerow(
                    [
                        name,
                        window,
                        trial,
                        raw["pearson"][trial],
                        raw["spearman"][trial],
                        raw["order_agreement"][trial],
                        raw["order_coverage"][trial],
                    ]
                )

    print(f"saved -> {out_md}")
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
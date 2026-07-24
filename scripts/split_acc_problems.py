#!/usr/bin/env python3
"""acc(TACO) 코퍼스를 problem_id 단위로 train/validation/test로 나눈다.

왜 별도 스크립트인가: 원본 덤프의 split은 critic별 행 확장 위에 얹혀 있어서 같은
problem_id가 여러 split에 걸친다(3,636건). 행 단위로 자르면 홀드아웃이 학습셋과
문제를 공유한다 — 실제로 이전 validation 500 중 210건(42%)이 train과 problem_id를
공유했고, 그 위에서 나온 조건 비교는 전부 무효였다.

여기서는 problem_id 유일성을 입구에서 강제하고, main_critic_category로 층화해서
나눈 뒤, 세 split이 problem_id를 하나도 공유하지 않는지 출구에서 다시 검증한다.

사용:
  python scripts/split_acc_problems.py export/acc_v2/acc_all_verified.jsonl \
      --out_dir export/acc_v2 --val 750 --test 750 --seed 20210111
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val", type=int, default=750, help="validation 문제 수")
    ap.add_argument("--test", type=int, default=750, help="test(최종 홀드아웃) 문제 수")
    ap.add_argument("--seed", type=int, default=20210111)
    ap.add_argument("--prefer_train", default=None,
                    help="여기 담긴 problem_id는 무조건 train으로 (binning 라벨 보유 문제). "
                         "라벨은 학습에 다 쓰고 홀드아웃은 라벨 없는 풀에서 뽑는다 — "
                         "두 풀의 카테고리/플랫폼 분포가 같은 것을 확인했다.")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    pids = [str(r.get("problem_id")) for r in rows]
    dup = [p for p, c in Counter(pids).items() if c > 1]
    if dup:
        raise SystemExit(f"problem_id 중복 {len(dup)}건 — 빌더에서 --dedupe-problem-id 없이 만든 파일이다")
    print(f"input={len(rows)} (problem_id 유일)")

    forced = set()
    if a.prefer_train:
        forced = {str(x) for x in json.load(open(a.prefer_train, encoding="utf-8"))}
        forced &= set(pids)
        print(f"prefer_train: {len(forced)}문제를 train에 고정 (홀드아웃 후보 {len(rows) - len(forced)})")

    by_cat = defaultdict(list)
    pinned = []
    for r in rows:
        if str(r["problem_id"]) in forced:
            pinned.append(r)
            continue
        by_cat[r.get("main_critic_category") or "UNKNOWN"].append(r)

    rng = random.Random(a.seed)
    val, test, train = [], [], list(pinned)
    total = sum(len(g) for g in by_cat.values())
    if a.val + a.test > total:
        raise SystemExit(f"홀드아웃 {a.val + a.test} > 후보 {total}")
    for cat in sorted(by_cat):
        g = sorted(by_cat[cat], key=lambda r: str(r["problem_id"]))
        rng.shuffle(g)
        # 층별 배분은 층 크기 비례 — 작은 층이 홀드아웃에서 통째로 빠지지 않게 한다.
        n_val = round(a.val * len(g) / total)
        n_test = round(a.test * len(g) / total)
        val += g[:n_val]
        test += g[n_val:n_val + n_test]
        train += g[n_val + n_test:]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "validation": val, "test": test}

    # 출구 검증: 세 split이 problem_id를 공유하면 안 된다.
    sets = {k: {str(r["problem_id"]) for r in v} for k, v in splits.items()}
    for x in ("train", "validation", "test"):
        for y in ("train", "validation", "test"):
            if x < y and sets[x] & sets[y]:
                raise SystemExit(f"{x}∩{y} = {len(sets[x] & sets[y])} — 분할 로직이 깨졌다")

    report = {"input": a.input, "seed": a.seed, "counts": {}, "category_distribution": {}}
    for name, g in splits.items():
        p = out / f"acc_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in sorted(g, key=lambda r: str(r["problem_id"])):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["counts"][name] = len(g)
        report["category_distribution"][name] = dict(
            Counter(r.get("main_critic_category") or "UNKNOWN" for r in g).most_common())
        print(f"  {name:11s} {len(g):6d} -> {p}")
    report["problem_id_disjoint"] = True
    json.dump(report, open(out / "split_report.json", "w"), ensure_ascii=False, indent=2)
    print(f"report -> {out / 'split_report.json'}")


if __name__ == "__main__":
    main()

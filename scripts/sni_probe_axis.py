"""SNI 프로브 판독 — 승패가 category 축을 따라 갈리나, domain 축을 따라 갈리나.

각 전문가는 자기 구역(strengths = category명 또는 sni_domain명)을 갖는다.
축이 실재한다면 전문가는 **자기 구역에서 남들보다 잘해야** 한다.
   home advantage = (자기 구역에서 내 pass율) − (자기 구역에서 나머지 전원 평균 pass율)
축이 없으면 이 값이 0 근처에서 흩어진다.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/sni/probe_raw.jsonl")
    ap.add_argument("--data", default="export/sni/sni_all.jsonl")
    ap.add_argument("--roster", default="configs/roster_sni_probe.json")
    a = ap.parse_args()

    meta = {}
    for line in open(a.data):
        r = json.loads(line)
        meta[r["id"]] = (r["category"], r["sni_domain"])

    roster = json.loads(Path(a.roster).read_text())
    region = {p["id"]: p["strengths"] for p in roster if p["id"] != "luca"}
    kind = {p["id"]: ("category" if p["id"].startswith("cat_") else "domain")
            for p in roster if p["id"] != "luca"}

    # (expert, pid) -> [pass...]  K회 평균
    acc: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for line in open(a.raw):
        d = json.loads(line)
        acc[(d["cid"], d["pid"])].append(d["pass"])
    phat = {k: sum(v) / len(v) for k, v in acc.items()}
    experts = sorted({k[0] for k in phat})
    pids = sorted({k[1] for k in phat})

    def region_items(reg: str, axis: str) -> list[str]:
        idx = 0 if axis == "category" else 1
        return [p for p in pids if p in meta and meta[p][idx] == reg]

    print(f"전문가 {len(experts)}명 · 문제 {len(pids)}개\n")
    print(f"{'expert':22s} {'구역':30s} {'n':>4s} {'자기':>7s} {'남들':>7s} {'차이':>8s} {'순위':>7s}")
    print("-" * 92)
    rows = []
    for e in experts:
        if e not in region:
            continue
        reg, ax = region[e], kind[e]
        items = region_items(reg, ax)
        if not items:
            continue
        mine = st.mean(phat[(e, p)] for p in items)
        others = [o for o in experts if o != e]
        oth = st.mean(st.mean(phat[(o, p)] for p in items) for o in others)
        # 그 구역에서 내 순위 (1=최고)
        scores = sorted(((st.mean(phat[(o, p)] for p in items), o) for o in experts), reverse=True)
        rank = [o for _, o in scores].index(e) + 1
        rows.append((ax, e, reg, len(items), mine, oth, mine - oth, rank))
        print(f"{e:22s} {reg[:28]:30s} {len(items):4d} {100*mine:6.1f}% {100*oth:6.1f}% "
              f"{100*(mine-oth):+7.2f}pp {rank:3d}/{len(experts)}")

    print("\n" + "=" * 92)
    for ax in ("category", "domain"):
        sub = [r for r in rows if r[0] == ax]
        if not sub:
            continue
        adv = [r[6] for r in sub]
        ranks = [r[7] for r in sub]
        wins = sum(1 for r in sub if r[6] > 0)
        print(f"{ax:9s} 전문가 {len(sub):2d}명 | home advantage 평균 {100*st.mean(adv):+6.2f}pp "
              f"(중앙값 {100*st.median(adv):+6.2f}pp) | 양수 {wins}/{len(sub)} "
              f"| 자기구역 평균순위 {st.mean(ranks):.1f}/{len(experts)} (우연 기대 {(len(experts)+1)/2:.1f})")

    # 구역별로 실제 1등이 누구인가
    print("\n구역별 실제 1등 (의도한 전문가가 이기는가):")
    hit = 0
    tot = 0
    for ax in ("category", "domain"):
        for e, reg in sorted(((e, r) for e, r in region.items() if kind[e] == ax), key=lambda x: x[1]):
            items = region_items(reg, ax)
            if not items:
                continue
            best = max(experts, key=lambda o: st.mean(phat[(o, p)] for p in items))
            tot += 1
            hit += best == e
            mark = "  <-- 적중" if best == e else ""
            print(f"  [{ax[:3]}] {reg[:30]:32s} 1등={best:22s}{mark}")
    print(f"\n적중 {hit}/{tot}  (우연 기대 {tot/len(experts):.1f})")


if __name__ == "__main__":
    main()

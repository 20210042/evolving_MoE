"""SNI 프로브용 구역별 균등 표집.

무작위로 뽑으면 꼬리 구역이 6문제씩밖에 안 잡혀 전문가별 검정력이 안 나온다
(300문제 기준 12번째 category 기대값 ≈ 6). 로스터의 각 전문가가 자기 구역에서
충분한 표본을 갖도록 균등하게 뽑는다.

구역 = 로스터 전문가의 strengths (category 이름 또는 sni_domain 이름).
한 item은 category 구역과 domain 구역에 동시에 속하므로, 가장 덜 채워진 구역부터
채우면서 소속된 모든 구역의 카운트를 올린다(= 같은 item이 두 구역을 동시에 채움).
같은 task에서 몰아 뽑히지 않게 task 단위로도 라운드로빈한다.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="export/sni/sni_all.jsonl")
    ap.add_argument("--roster", default="configs/roster_sni_probe.json")
    ap.add_argument("--n", type=int, default=600, help="총 문제 수 상한")
    ap.add_argument("--per-region", type=int, default=27, help="구역당 목표 표본")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sni/probe_problem_ids.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(a.data)]
    roster = json.loads(Path(a.roster).read_text())
    regions = [p["strengths"] for p in roster if p["id"] != "luca"]

    # item -> 소속 구역들
    members: dict[str, list[int]] = {r: [] for r in regions}
    belongs: list[list[str]] = []
    for i, row in enumerate(rows):
        rs = [r for r in (row.get("category"), row.get("sni_domain")) if r in members]
        belongs.append(rs)
        for r in rs:
            members[r].append(i)

    for r in regions:
        rng.shuffle(members[r])
        print(f"  구역 {r[:34]:36s} 풀 {len(members[r]):6,}")

    # task 단위 라운드로빈용: 구역 안에서 task를 번갈아
    bytask: dict[str, dict[str, list[int]]] = {}
    for r in regions:
        d = collections.defaultdict(list)
        for i in members[r]:
            d[rows[i]["task_name"]].append(i)
        order = list(d)
        rng.shuffle(order)
        bytask[r] = {"order": order, "d": d, "p": 0}

    filled: collections.Counter = collections.Counter()
    chosen: list[int] = []
    used: set[int] = set()

    while len(chosen) < a.n:
        cand = [r for r in regions if filled[r] < a.per_region and bytask[r]["order"]]
        if not cand:
            break
        r = min(cand, key=lambda x: filled[x])
        st = bytask[r]
        pick = None
        for _ in range(len(st["order"])):
            t = st["order"][st["p"] % len(st["order"])]
            st["p"] += 1
            pool = st["d"][t]
            while pool and pool[-1] in used:
                pool.pop()
            if pool:
                pick = pool.pop()
                break
        if pick is None:
            st["order"] = []
            continue
        used.add(pick)
        chosen.append(pick)
        for rr in belongs[pick]:
            filled[rr] += 1

    ids = [rows[i]["id"] for i in chosen]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(ids, ensure_ascii=False, indent=1))

    print(f"\n총 {len(ids)} 문제 / task {len({rows[i]['task_name'] for i in chosen})}종")
    print("\n구역별 확보 표본:")
    for r in regions:
        print(f"  {r[:34]:36s} {filled[r]:3d}")
    short = [r for r in regions if filled[r] < a.per_region]
    if short:
        print(f"\n목표 미달 구역: {short}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()

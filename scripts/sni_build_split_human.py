#!/usr/bin/env python
"""대조군 분할 — Human prior split, **순수 BTX식**. 재생성 0.

BTX(Branch-Train-MiX)가 실제로 하는 것: 학습 데이터를 **겹치지 않는 도메인으로 나누고**
각 expert가 자기 도메인만 학습한다. 그대로 따른다.

  · shared expert **없음** (16 routed only)
  · 한 문제는 정확히 **1명**에게 간다 (중복 없음 → 학습 row = train 문제 수)
  · 진화 쪽 n_solved 구간(indiv/shared/all_fail)은 **쓰지 않는다** — teacher가 만든 사실이라
    사람 사전지식 조건에 넣으면 그 축이 새어 들어온다
  · 축은 SNI 공식 `category` 72개

⚠️ "상위 16개 category만 쓴다"는 안 된다 — top-16이 68.0%라 나머지 56개 22,236건(32%)이
   버려져 다른 조건과 데이터량이 달라진다. 그래서 **72개 category를 겹치지 않는 16그룹으로 묶는다**:
   크기 내림차순으로 훑으며 그때까지 가장 작은 그룹에 넣는다(그리디 LPT).
   **category는 절대 쪼개지 않는다** — 각 expert는 온전한 category 집합만 갖는다.
   그 결과 expert별 학습량은 고르지 않다(최대 category 하나가 평균 정원보다 크다).
   그 불균형은 보정 대상이 아니라 **사람 택소노미의 성질 자체**다.

expert 슬롯 id는 다른 조건과 파일 형식을 맞추기 위해 로스터 id를 재사용할 뿐,
페르소나와 아무 관계가 없다. 각 슬롯이 실제로 맡은 category는 리포트와 manifest에 적는다.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

TRAIN = "export/sni_v4/sni_train.jsonl"
ROSTER = "results/sni/seed20212003/roster_final.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="category", choices=["category", "sni_domain"])
    ap.add_argument("--n_experts", type=int, default=16)
    ap.add_argument("--out", default="export/sni_split_human/split.jsonl")
    ap.add_argument("--report", default="results/sni/split_build_human.md")
    a = ap.parse_args()

    ids, axis = [], {}
    for l in open(TRAIN):
        d = json.loads(l)
        ids.append(d["id"])
        axis[d["id"]] = d.get(a.axis)
    size = Counter(axis.values())
    E = a.n_experts

    # 그리디 LPT — 큰 category부터 그때까지 가장 작은 그룹에 넣는다. category는 쪼개지 않는다.
    grp = [[] for _ in range(E)]
    tot = [0] * E
    for c in sorted(size, key=lambda c: (-size[c], str(c))):
        j = min(range(E), key=lambda j: (tot[j], j))
        grp[j].append(c)
        tot[j] += size[c]

    # 슬롯 id는 로스터 id 재사용(파일 형식 통일용). 큰 그룹부터 붙인다.
    ex = [p["id"] for p in json.load(open(ROSTER))]
    assert len(ex) >= E, (len(ex), E)
    order = sorted(range(E), key=lambda j: -tot[j])
    cat2ex = {c: ex[k] for k, j in enumerate(order) for c in grp[j]}
    members = defaultdict(list)
    for c, e in cat2ex.items():
        members[e].append(c)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cnt = Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for pid in ids:
            e = cat2ex[axis[pid]]
            cnt[e] += 1
            f.write(json.dumps({"id": pid, "kind": "btx", "experts": [e],
                                "n_solved": -1}, ensure_ascii=False) + "\n")

    N = len(ids)
    L = [f"# 대조군 분할 — Human prior split, 순수 BTX식 (축={a.axis}, E={E})", "",
         "BTX 그대로다: 데이터를 **겹치지 않는 도메인으로 나누고** 각 expert가 자기 도메인만 학습한다.",
         "**shared expert 없음 · 문제당 1명 · 중복 없음 · 진화 쪽 n_solved 구간 미사용.**", "",
         f"- train {N:,}문제 전수 → 학습 row {sum(cnt.values()):,} (중복이 없으므로 문제 수와 같다)",
         f"- 축값 {len(size)}개를 겹치지 않는 {E}그룹으로 묶었다 (그리디 LPT, category 미분할)",
         f"- expert별 학습량 최대 {max(cnt.values()):,} / 최소 {min(cnt.values()):,} "
         f"(균등이면 {N//E:,}) — 최대 축값 `{max(size, key=size.get)}` "
         f"{size[max(size, key=size.get)]:,}건이 통째로 한 명에게 가서 생기는 편차이고, "
         "**사람 택소노미의 성질이라 보정하지 않는다**", "",
         "| # | expert 슬롯 | 학습 row | 맡은 축값 수 | 맡은 축값 |",
         "|---:|---|---:|---:|---|"]
    for i, e in enumerate(ex[:E]):
        m = sorted(members[e], key=lambda c: -size[c])
        shown = ", ".join(f"{c} ({size[c]:,})" for c in m[:6])
        if len(m) > 6:
            shown += f", … 외 {len(m)-6}개"
        L.append(f"| {i} | {e} | **{cnt[e]:,}** | {len(m)} | {shown} |")
    L += ["", "expert 슬롯 id는 파일 형식을 다른 조건과 맞추려고 로스터 id를 재사용한 것뿐이고 "
              "**페르소나와 아무 관계가 없다**.", "", f"산출: `{a.out}`"]
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    open(a.report, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

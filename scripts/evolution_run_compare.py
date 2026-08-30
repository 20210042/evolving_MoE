#!/usr/bin/env python3
"""세 진화 런 비교 + 실행 로그 기반 재현성(온라인 버전).

seed20210111(hard-WAR, gatefix) vs seed20211001(soft_linear, 장치 없음) vs
seed20211002(soft_partial, 장치 복원)의 로스터 궤적을 비교하고, 각 런의 실제 evolution_log.jsonl
에 남은 war 점수(스텝마다 실제 생성으로 계산된 값)를 써서 "연속 스텝 간 최하위 지목이
재현되는가"를 오프라인 스크리닝(soft_war_screen.py, 문제 절반 나누기)과 별개로 다시 잰다 —
이번엔 실제 실행 데이터, split-half 아니라 인접 스텝 쌍(temporal pair)이 표본.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "20210111 (hard-WAR, gatefix)": "results/acc/seed20210111/acc/seed20210111",
    "20211001 (soft_linear, 장치없음)": "results/acc/seed20211001/acc/seed20211001",
    "20211002 (soft_partial, 장치복원)": "results/acc/seed20211002/acc/seed20211002",
}
OUT = ROOT / "results/acc/evolution_run_compare.md"
L: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    L.append(s)


def load_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(ROOT / path / "evolution_log.jsonl", encoding="utf-8")]


def dirchanges(seq: list[int]) -> int:
    d = [b - a for a, b in zip(seq, seq[1:]) if b != a]
    return sum(1 for a, b in zip(d, d[1:]) if (a > 0) != (b > 0))


say("# 세 진화 런 비교 — 궤적 + 실행 로그 기반 재현성")
say()
say("## 1. 로스터 궤적")
say()
say("| 런 | 스텝 | 크기 min/max/final | 방향전환 | add/delete/swap | 후반30% UB평균 |")
say("|---|---:|---|---:|---|---:|")
summaries = {}
for name, path in RUNS.items():
    rows = load_rows(path)
    sizes = [len(r["roster_after"]) for r in rows]
    ub = [r["upper_bound_pct"] for r in rows]
    dec = [r["decision"] for r in rows]
    cnt = {k: dec.count(k) for k in ("add", "delete", "swap")}
    tail_ub = st.mean(ub[-max(1, len(ub) // 3):])
    say(f"| {name} | {len(rows)} | {min(sizes)}/{max(sizes)}/{sizes[-1]} | "
        f"{dirchanges(sizes)} | {cnt['add']}/{cnt['delete']}/{cnt['swap']} | {tail_ub:.1f}% |")
    summaries[name] = (rows, sizes)
say()
say("**20211002는 방향전환 0 — 삭제가 단 한 번도 안 일어났다.** '안정'이 아니라 삭제 게이트가")
say("사실상 작동을 멈춘 것이다(§3에서 원인 확인).")
say()

say("## 2. 실행 로그 기반 재현성 (인접 스텝 쌍, 로스터 크기가 안정된 구간만)")
say()
say("오프라인 스크리닝(`soft_war_screen.md`)은 10,000문제를 인위적으로 반씩 갈랐다.")
say("여기서는 **실제 생성으로 나온 스텝별 war 점수**를 그대로 쓴다 — 스텝 t와 t+1을 한 쌍으로 보고,")
say("t에서 최하위였던 사람이 t+1에서도 최하위인지 잰다. 로스터 크기가 바뀌면 비교가 안 되므로")
say("**크기가 가장 오래 유지된 구간**만 쓴다.")
say()
for name, path in RUNS.items():
    rows, sizes = summaries[name]
    # 최빈 크기의 최장 연속 구간 찾기
    from collections import Counter
    common = Counter(sizes).most_common(1)[0][0]
    best_run = []
    cur = []
    for i, s in enumerate(sizes):
        if s == common:
            cur.append(i)
        else:
            if len(cur) > len(best_run):
                best_run = cur
            cur = []
    if len(cur) > len(best_run):
        best_run = cur
    idx = best_run
    if len(idx) < 3:
        say(f"**{name}**: 안정 구간이 너무 짧음(<3스텝) — 생략")
        say()
        continue
    war_seq = [rows[i]["war"] for i in idx]
    members = set(war_seq[0])
    for w in war_seq:
        members &= set(w)
    members = sorted(members)
    n = len(members)
    same_loser, unique_min_pairs, same_loser_unique = 0, 0, 0
    total_pairs, corrs, tie_sizes = 0, [], []
    for a, b in zip(war_seq, war_seq[1:]):
        va = [a[m] for m in members]
        vb = [b[m] for m in members]
        if len(set(va)) < 2 or len(set(vb)) < 2:
            continue
        mv = min(va); mv2 = min(vb)
        loser_a = {m for m, v in zip(members, va) if v == mv}
        loser_b = {m for m, v in zip(members, vb) if v == mv2}
        tie_sizes.append(len(loser_a))
        total_pairs += 1
        if loser_a & loser_b:
            same_loser += 1
        # 동점을 반영한 공정한 비교: a쪽 최하위가 "유일"했던 경우만 따로 집계
        # (동점 그룹이 크면 교집합이 우연으로도 잘 생겨 재지목률이 뻥튀기된다)
        if len(loser_a) == 1:
            unique_min_pairs += 1
            if loser_a & loser_b:
                same_loser_unique += 1
        ma, mb = sum(va) / n, sum(vb) / n
        sa = (sum((x - ma) ** 2 for x in va)) ** 0.5
        sb = (sum((x - mb) ** 2 for x in vb)) ** 0.5
        if sa > 1e-9 and sb > 1e-9:
            cov = sum((x - ma) * (y - mb) for x, y in zip(va, vb))
            corrs.append(cov / (sa * sb))
    r_mean = st.mean(corrs) if corrs else float("nan")
    say(f"**{name}** — 안정 구간: 로스터 크기 {common}명 × {len(idx)}스텝 (전문가 {n}명 교집합)")
    say(f"- 인접 스텝 간 점수 상관: **{r_mean:+.3f}** (n={len(corrs)}쌍)")
    say(f"- 스텝별 최소값 동점 그룹 평균 크기: **{st.mean(tie_sizes):.1f}/{n}명** "
        f"(크면 아래 재지목률이 우연으로도 부풀려진다)")
    say(f"- 인접 스텝 최하위 재지목률(동점 포함, 원판정): **{100*same_loser/total_pairs:.0f}%** "
        f"(n={total_pairs}쌍, 무작위 기대 ≈ {100/n:.0f}%)")
    if unique_min_pairs >= 5:
        say(f"- **동점 없이 유일한 최하위였던 경우만**: **{100*same_loser_unique/unique_min_pairs:.0f}%** "
            f"(n={unique_min_pairs}쌍) ← 더 공정한 비교")
    else:
        say(f"- 동점 없는 유일한 최하위 사례가 {unique_min_pairs}건뿐이라 별도 집계 생략")
    say()

say("## 2-1. 배치 하나로는 노이즈에 묻힌다 — 윈도로 누적하면 나아지는가")
say()
say("사전 오프라인 검증(`soft_war_screen.md`)은 문제 **5,000개씩** 절반으로 갈라 79%가 나왔다.")
say("§2는 배치 하나(100문제) vs 다음 배치(100문제)라 표본이 50배 작다 — 그래서 낮게 나왔을 수 있다.")
say("`deletion_window`(=16)만큼 최근 배치의 soft_partial 점수를 **누적 평균**해서 같은 검정을 다시 한다.")
say()
WIN = 16
for name, path in RUNS.items():
    if "20211002" not in name:
        continue
    rows, sizes = summaries[name]
    from collections import Counter
    common = Counter(sizes).most_common(1)[0][0]
    idx = [i for i, s in enumerate(sizes) if s == common]
    # 연속 구간 중 가장 긴 것 (윈도 누적을 위해 연속성 필요)
    runs_, cur = [], []
    for i in range(len(sizes)):
        if sizes[i] == common:
            cur.append(i)
        else:
            if cur:
                runs_.append(cur)
            cur = []
    if cur:
        runs_.append(cur)
    idx = max(runs_, key=len)
    if len(idx) < WIN + 5:
        say(f"**{name}**: 안정 구간이 윈도+5보다 짧아 생략")
        continue
    war_seq = [rows[i]["war"] for i in idx]
    members = set(war_seq[0])
    for w in war_seq:
        members &= set(w)
    members = sorted(members)
    n = len(members)
    # 시점 t의 "누적 점수" = war_seq[t-WIN:t] 평균
    acc = []
    for t in range(WIN, len(war_seq)):
        window = war_seq[t - WIN:t]
        avg = {m: sum(w[m] for w in window) / WIN for m in members}
        acc.append(avg)
    same_loser, total_pairs, corrs = 0, 0, []
    for a, b in zip(acc, acc[1:]):
        va = [a[m] for m in members]; vb = [b[m] for m in members]
        if len(set(va)) < 2 or len(set(vb)) < 2:
            continue
        mv = min(va); mv2 = min(vb)
        la = {m for m, v in zip(members, va) if v == mv}
        lb = {m for m, v in zip(members, vb) if v == mv2}
        total_pairs += 1
        if la & lb:
            same_loser += 1
        ma, mb = sum(va) / n, sum(vb) / n
        sa = (sum((x - ma) ** 2 for x in va)) ** 0.5
        sb = (sum((x - mb) ** 2 for x in vb)) ** 0.5
        if sa > 1e-9 and sb > 1e-9:
            corrs.append(sum((x - ma) * (y - mb) for x, y in zip(va, vb)) / (sa * sb))
    r_mean = st.mean(corrs) if corrs else float("nan")
    say(f"**{name}** — {WIN}배치 누적, 연속구간 {len(idx)}스텝 → 비교쌍 {total_pairs}개")
    say(f"- 누적 점수 인접 상관: **{r_mean:+.3f}** (단발 배치 대비 §2 참고)")
    say(f"- 누적 최하위 재지목률: **{100*same_loser/max(1,total_pairs):.0f}%**")
    say()

say("## 3. 왜 20211002는 삭제가 0번인가 — 회복이 손실을 거의 매번 지운다")
say()
say("`lives_mode=rank`의 회복은 `lives = min(max_lives, lives+1)`이다. 최하위로 한 번 찍혀")
say("5→4가 돼도, 바로 다음 스텝에 안 찍히면 그걸로 다시 5로 **완전히 원상복구**된다.")
say("`max_lives=5`이므로 0까지 가려면 **같은 사람이 5번 연달아** 찍혀야 하는데,")
say("실제 로그(잡 221549+222433, 111스텝 전체)를 세어보면:")
say()
say("| | 값 |")
say("|---|---:|")
say("| 인접 두 스텝이 같은 사람을 지목한 횟수(=진짜 순감점 발생) | **6 / 110쌍**")
say("| 그중 3연속 이상으로 이어진 경우 | **0건**")
say()
say("11명 중 매 스텝 1명이 뽑히는 사건(≈9%)이 같은 사람에게 5번 연달아 몰릴 확률은")
say("사실상 0에 가깝다 — 로스터 크기와 스텝 수 대비 이 설계로는 삭제가 발동할 수 없었다.")
say("최종 로스터 11명 중 10명이 lives=5(만땅), 1명만 lives=4. 최하위 지목 자체는")
say("5~17회씩 골고루 있었지만(특정 소수에 안 몰림) 전부 다음 스텝 회복으로 상쇄됐다.")
say()
say("**→ '로스터가 안정됐다'가 아니라 '삭제 게이트가 이 설정에서 작동 불가능했다'가 정확한 해석이다.**")
say("소프트 신호가 좋은지 나쁜지는 이 런만으로는 아직 검증되지 않았다 — 삭제가 한 번도 없었으니")
say("최하위 판정이 실제로 옳은 사람을 내보내는지 관찰할 기회 자체가 없었다.")
say()

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"saved -> {OUT}")

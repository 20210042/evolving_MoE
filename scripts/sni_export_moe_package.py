#!/usr/bin/env python
"""협업자 인계용 16+1 expert 학습 패키지 생성. 재생성 0.

입력  : export/sni_split_seed20212003/split.jsonl (배정) + export/sni_v4/sni_{train,test}.jsonl (원본)
출력  : export/sni_moe_seed20212003/

배정 규칙(이미 확정, sni_build_split.py):
  1 ≤ n_solved ≤ 10 → 푼 사람 전원에게 개별 배정 (한 문제가 여러 expert에 중복 등장한다)
  n_solved > 10     → shared expert (전원 성공 포함)
  n_solved = 0      → 센트로이드(tc=5 가중) 최근접 expert 1명

프롬프트: **페르소나 없음**. system = SNI_GEN_SYSTEM + 태스크 정의, user = 공식 Tk-Instruct
형식(positive example 2건 + Input/Output). 페르소나는 teacher가 분할을 만들 때만 썼고
student 학습에는 넣지 않는다 — 그게 이 실험의 조건이다.
학습 타깃 = **gold**(페르소나 출력 아님).

⚠️ ground_truth는 리스트다(SNI는 복수 정답 허용, 공식 채점은 레퍼런스 최대값).
   SFT는 문자열 하나가 필요하므로 `target`에 **첫 레퍼런스**를 쓰고, 전체는 `targets`로 보존한다.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import router_common as rc  # noqa: E402
import prompts.baseline_prompts as bp  # noqa: E402
from prompts.coding import (build_baseline_prompt, sni_system_block,  # noqa: E402
                            sni_user_block)

OUT = Path("export/sni_moe_seed20212003")
SPLIT = "export/sni_split_seed20212003/split.jsonl"
ARM = "ours"
DATASET = "sni"
SEED = "seed20212003"
STUDENT = "meta-llama/Llama-3.1-8B"
ROSTER = "results/sni/seed20212003/roster_final.json"
SHARED = "__shared__"

# ⚠️ 조건마다 배정 규칙이 다르다. 구간(indiv/shared/all_fail)과 문제당 인원 수,
#    expert별 총량은 세 조건이 전부 같고 **누가 맡느냐만** 다르다.
_BANDS = "n_solved 기준 (1..10 개별 · >10 shared · 0 전원실패) — 세 조건 공통"
SCORING = {
    "sni": "EM==100 or ROUGE-L>70 (src/evaluation/scorer.py: score_sni_item)",
    "lbox": "정답 일치 (src/evaluation/scorer.py: score_lbox_item) — "
            "casename은 정규화 후 완전일치, statute는 gold 집합 대조",
}
SPLIT_RULE = {
    "ours": {"bands": _BANDS, "axis": "진화 로스터의 실제 solve",
             "tau_train": 10, "tau_centroid": 5,
             "centroid_weight": "(E-n)/(E-1)",
             "feature": "hs_mean (teacher 마지막 층 평균 풀링, 차원별 z-정규화 후 L2)",
             "indiv": "푼 expert 전원", "all_fail": "센트로이드 최근접 1명",
             "all_pass": "shared", "builder": "scripts/sni_build_split.py"},
    "random": {"bands": _BANDS, "axis": "무작위 (대조군)",
               "assignment": "Ours의 expert별 총량을 정원으로 두고 차수 보존 무작위 재배치",
               "quota_source": "Ours expert별 총량 (kind별)", "seed": 0,
               "all_pass": "shared", "builder": "scripts/sni_build_split_random.py"},
    "human": {"bands": "없음 — 진화 쪽 n_solved 구간을 쓰지 않는다 (사람 사전지식 조건)",
              "axis": "사람 택소노미 — SNI 공식 category 72개 (대조군)",
              "style": "순수 BTX (Branch-Train-MiX) — 겹치지 않는 도메인 분할",
              "assignment": "category 72개를 겹치지 않는 16그룹으로 묶는다"
                            " (크기 내림차순 그리디 LPT, category는 쪼개지 않는다)."
                            " 한 문제는 자기 category를 맡은 expert 1명에게만 간다",
              "shared_expert": False, "duplication": "없음 (학습 row = train 문제 수)",
              "coverage": "train 전수 69,588 — 상위 16 category만 쓰면 32%가 버려져 쓰지 않았다",
              "imbalance": "expert별 12,499~3,500 — 최대 category가 통째로 한 명에게 가서 생기는"
                           " 편차이고 사람 택소노미의 성질이라 보정하지 않는다",
              "builder": "scripts/sni_build_split_human.py"},
}


def build(r):
    """(system, user) 조립. 도메인마다 baseline GEN 프롬프트가 다르다 — 페르소나는 없다."""
    if DATASET == "sni":
        return (sni_system_block(bp.SNI_GEN_SYSTEM, r.get("definition")),
                sni_user_block(r.get("answer_line"), r["instruction"],
                               positive_examples=r.get("positive_examples"), num_pos=2))
    # LBox/QASC 등은 build_baseline_prompt가 도메인별 GEN 쌍을 준다
    # (2026-07-13 합의: per-expert SFT는 baseline GEN으로 통일, 페르소나 금지).
    msgs = build_baseline_prompt(r["instruction"], dataset=DATASET,
                                 model_name=STUDENT, domain=r.get("domain"))
    return msgs[0]["content"], msgs[1]["content"]


def row(r, expert, kind, n_solved):
    sysm, usr = build(r)
    gt = r.get("ground_truth")
    refs = [str(g) for g in gt] if isinstance(gt, list) else ([str(gt)] if gt is not None else [])
    d = {"id": r["id"], "expert": expert, "kind": kind, "n_solved": n_solved,
         "system": sysm, "user": usr,
         "target": refs[0] if refs else "", "targets": refs}
    if DATASET == "sni":
        d.update({"task_name": r.get("task_name"), "category": r.get("category"),
                  "sni_domain": r.get("sni_domain"), "task_closed": r.get("task_closed")})
    else:   # LBox: 사건 유형·설정을 남긴다(분석용, 학습에는 안 쓴다)
        d.update({"task_type": r.get("task_type"), "task_config": r.get("task_config"),
                  "casetype": r.get("casetype")})
    return d


def main():
    global OUT, SPLIT, ARM
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ours", help="ours | random | human")
    ap.add_argument("--dataset", default="sni", help="sni | lbox")
    ap.add_argument("--roster", default=None)
    ap.add_argument("--tau", type=int, default=10, help="manifest 기록용 (분할 빌더와 같은 값)")
    ap.add_argument("--tau_c", type=int, default=5, help="manifest 기록용")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global DATASET, ROSTER, SEED
    DATASET = a.dataset
    ARM = a.arm
    SPLIT = a.split or SPLIT
    OUT = Path(a.out or OUT)
    sp = rc.spec(DATASET)
    ROSTER = a.roster or ROSTER
    m = re.search(r"seed\d+", ROSTER)      # 시드는 로스터 경로에서 읽는다
    SEED = m.group(0) if m else SEED
    SPLIT_RULE["ours"].update({"tau_train": a.tau, "tau_centroid": a.tau_c})
    if DATASET != "sni":
        SPLIT_RULE["ours"]["feature"] = (
            "embed_viz 임베딩 (차원별 z-정규화 후 L2). "
            "⚠️ SNI는 teacher 마지막 층 hs_mean이었다 — 임베딩 종류가 다르다")
    src = {json.loads(l)["id"]: json.loads(l) for l in open(sp.src["train"])}
    assign = {}
    for l in open(SPLIT):
        d = json.loads(l)
        assign[d["id"]] = d
    roster = json.load(open(ROSTER))
    ex = [p["id"] for p in roster]
    names = {p["id"]: p["name"] for p in roster}
    persona = {p["id"]: p.get("system_prompt") or p.get("prompt") for p in roster}

    (OUT / "train").mkdir(parents=True, exist_ok=True)
    has_shared = any(SHARED in d["experts"] for d in assign.values())
    fh = {c: open(OUT / "train" / f"expert_{i:02d}_{c}.jsonl", "w", encoding="utf-8")
          for i, c in enumerate(ex)}
    if has_shared:
        fh[SHARED] = open(OUT / "train" / "shared.jsonl", "w", encoding="utf-8")
    else:   # 순수 BTX 조건에는 shared expert가 없다. 이전 실행의 잔재를 남기지 않는다.
        (OUT / "train" / "shared.jsonl").unlink(missing_ok=True)
    cnt = Counter()
    kind_cnt = {c: Counter() for c in list(ex) + [SHARED]}
    cat_of = {c: Counter() for c in list(ex) + [SHARED]}
    miss = 0
    for pid, d in assign.items():
        r = src.get(pid)
        if r is None:
            miss += 1
            continue
        for c in d["experts"]:
            fh[c].write(json.dumps(row(r, c, d["kind"], d["n_solved"]), ensure_ascii=False) + "\n")
            cnt[c] += 1
            kind_cnt[c][d["kind"]] += 1
            cat_of[c][r.get("category")] += 1
    for f in fh.values():
        f.close()

    # 평가셋: SNI는 test 하나, LBox는 파이프라인 기준이 valid라 valid·test 둘 다 낸다.
    eval_splits = ["test"] if DATASET == "sni" else ["valid", "test"]
    n_test = 0
    n_eval = {}
    for esp in eval_splits:
        with open(OUT / f"{esp}.jsonl", "w", encoding="utf-8") as f:
            n = 0
            for l in open(sp.src[esp]):
                r = json.loads(l)
                f.write(json.dumps(row(r, None, esp, -1), ensure_ascii=False) + "\n")
                n += 1
        n_eval[esp] = n
    n_test = n_eval[eval_splits[-1]]

    experts = [{"index": i, "id": c, "name": names[c] if ARM == "ours" else f"expert {i}",
                "n_train": cnt[c], "n_by_kind": dict(kind_cnt[c]),
                "n_indiv": kind_cnt[c]["indiv"], "n_all_fail": kind_cnt[c]["all_fail"],
                "categories": sorted(cat_of[c], key=lambda k: -cat_of[c][k]) or None,
                "persona_system_prompt_TEACHER_ONLY": persona[c] if ARM == "ours" else None}
               for i, c in enumerate(ex)]
    if has_shared:
        experts.append({"index": len(ex), "id": SHARED, "name": "shared expert",
                        "n_train": cnt[SHARED], "n_by_kind": dict(kind_cnt[SHARED]),
                        "n_indiv": cnt[SHARED], "n_all_fail": 0, "categories": None,
                        "persona_system_prompt_TEACHER_ONLY": None})
    manifest = {
        "seed": SEED, "arm": ARM, "dataset": DATASET, "n_experts_routed": len(ex),
        "shared_expert": has_shared, "teacher": "google/gemma-4-26B-A4B-it",
        "student_intended": "meta-llama/Llama-3.1-8B",
        "split_rule": SPLIT_RULE[ARM],
        "target": "gold (ground_truth 첫 레퍼런스; 전체는 targets)",
        "prompt": "페르소나 없음 · system=SNI_GEN_SYSTEM+definition · user=공식 Tk-Instruct(pos 2건)",
        "n_train_problems": len(assign), "n_train_rows": sum(cnt.values()),
        "n_eval": n_eval, "n_test": n_test, "source_missing": miss,
        "scoring": SCORING[DATASET],
        "experts": experts,
    }
    json.dump(manifest, open(OUT / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    L = [f"# {DATASET.upper()} {len(ex)}{'+1' if has_shared else ''} expert 학습 패키지 "
         f"({SEED}, arm={ARM})", "",
         f"- train 문제 {len(assign):,} → 학습 row {sum(cnt.values()):,} "
         f"(개별 배정 구간은 한 문제가 여러 expert에 중복 등장한다)",
         "- 평가셋(배정 없음, 조건 공통): " +
         " · ".join(f"`{k}.jsonl` {v:,}" for k, v in n_eval.items()),
         f"- 원본에서 못 찾은 id: {miss}", "",
         "## 파일", "",
         "| 파일 | 내용 |", "|---|---|",
         "| `manifest.json` | 배정 규칙·expert 목록·행 수. 여기부터 읽으면 된다 |",
         f"| `train/expert_00..{len(ex)-1:02d}_<id>.jsonl` | routed expert {len(ex)}명의 학습 데이터 |",
         ("| `train/shared.jsonl` | shared expert(짬통) 학습 데이터 |" if has_shared
          else "| (shared 없음) | 순수 BTX 조건이라 shared expert가 없다 |"),
         "| `<split>.jsonl` | 공통 평가셋 (같은 프롬프트 형식) |", "",
         "## row 스키마", "",
         "```json",
         f'{{"id":..., "expert":..., "kind":"indiv|shared|all_fail", "n_solved":0-{len(ex)},',
         ' "system":"...", "user":"...", "target":"gold 문자열", "targets":["gold 전체"],',
         (' "task_name":..., "category":..., "sni_domain":..., "task_closed":...}' if DATASET == "sni"
          else ' "task_type":..., "task_config":..., "casetype":...}'),
         "```", "",
         "`system`/`user`는 이미 조립돼 있다. 그대로 chat template에 넣으면 된다.",
         "**페르소나는 들어 있지 않다** — teacher가 분할을 만들 때만 썼고 student 학습에는 넣지 않는다.",
         ("`manifest.json`의 `persona_system_prompt_TEACHER_ONLY`는 참고용이다."
          if ARM == "ours"
          else "expert 슬롯 id는 파일 형식을 다른 조건과 맞추려고 재사용한 것일 뿐 "
               "**페르소나와 아무 관계가 없다**."), "",
         "## expert별 학습량", "",
         ("| # | expert | 학습 row | 개별(n≤10) | 전원실패 |" if has_shared
          else "| # | expert 슬롯 | 학습 row | 맡은 category |"),
         "|---:|---|---:|---:|---:|" if has_shared else "|---:|---|---:|---|"]
    for e in experts:
        if has_shared:
            L.append(f"| {e['index']} | {e['name']} | **{e['n_train']:,}** | "
                     f"{e['n_indiv']:,} | {e['n_all_fail']:,} |")
        else:
            cs = e["categories"] or []
            shown = ", ".join(cs[:5]) + (f", … 외 {len(cs)-5}개" if len(cs) > 5 else "")
            L.append(f"| {e['index']} | {e['id']} | **{e['n_train']:,}** | {shown} |")
    L += ["", "## 채점", "", SCORING[DATASET]]
    if DATASET == "sni":
        L.append("공식(Tk-Instruct)은 임계 없이 EM·ROUGE를 병기한다. 이진 판정은 우리 쪽 요구사항이다.")
    L.append("")
    open(OUT / "README.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

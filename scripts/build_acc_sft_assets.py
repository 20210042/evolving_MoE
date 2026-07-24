#!/usr/bin/env python3
"""코딩(acc) MoL 사다리용 SFT 자산 구축 — seed20210111.

**타깃은 원본의 실행검증된 참조 솔루션이다.** 이전 판은 "acc에 참조 솔루션이 없다"는
잘못된 전제로 에이전트 생성코드를 타깃으로 승격시켰고(주석 라인 27%·`pass` 스텁 849개),
그 스타일을 학습한 모델이 홀드아웃에서 생성의 41%를 주석 무한루프로 태웠다.
참조 솔루션은 원본 덤프에 처음부터 있었다(97%가 주석 0줄) — build_acc_selfconsistent.py가
`--keep-solution`으로 실어 나른다.

QASC와 동일 논리 유지: **문제당 canonical 정답코드 1개**(= 검증된 refs[0]),
per_expert는 "어떤 문제를 보느냐"만 결정한다(타깃은 expert 무관 동일).

홀드아웃은 split_acc_problems.py가 problem_id 단위로 이미 갈라놨다 — 여기서 다시
자르지 않는다(이전 판이 행 단위 tail split로 42% 누수를 만들었다).

사용: python scripts/build_acc_sft_assets.py [--corpus_dir export/acc_v2] [--dup first|union]
"""
import argparse
import collections
import json
from pathlib import Path

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
SEED = "seed20210111"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", default="export/acc_v2",
                    help="split_acc_problems.py 산출물(acc_train/validation/test.jsonl)")
    ap.add_argument("--binned", default=f"results/acc/{SEED}/binning_train_full.binned.jsonl")
    ap.add_argument("--old_corpus", default="export/acc_selfconsistent/acc_train.jsonl",
                    help="binning 라벨의 id -> problem_id 매핑용(라벨은 구 코퍼스 id로 찍혀 있다)")
    ap.add_argument("--pkg", default=f"export/acc_binning_{SEED}_v2")
    ap.add_argument("--dup", choices=["first", "union"], default="first",
                    help="같은 problem_id가 구 코퍼스에서 여러 행으로 확장돼 두 번 풀린 경우(2,056건) "
                         "라벨 병합 규칙. first=정렬 첫 행(한 문제 한 번의 시도로 취급), "
                         "union=한 번이라도 풀었으면 solve(난이도를 낮게 보이게 만든다)")
    a = ap.parse_args()

    corpus = REPO / a.corpus_dir
    train_rows = [json.loads(l) for l in open(corpus / "acc_train.jsonl", encoding="utf-8")]
    by_pid = {str(r["problem_id"]): r for r in train_rows}
    print(f"코퍼스 train {len(train_rows)}문제 (참조 솔루션 보유)")

    pid_of = {}
    for l in open(REPO / a.old_corpus, encoding="utf-8"):
        r = json.loads(l)
        pid_of[str(r["id"])] = str(r["problem_id"])

    binned = [json.loads(l) for l in open(REPO / a.binned, encoding="utf-8")]
    EX = sorted(binned[0]["per_expert"])
    labels = {}
    for b in sorted(binned, key=lambda b: str(b["id"])):
        pid = pid_of.get(str(b["id"]))
        if pid is None:
            continue
        if pid not in labels:
            labels[pid] = {e: int(b["per_expert"].get(e, 0)) for e in EX}
        elif a.dup == "union":
            for e in EX:
                labels[pid][e] |= int(b["per_expert"].get(e, 0))
    print(f"binning 라벨 {len(binned)}행 -> {len(labels)}문제 × {len(EX)} experts (dup={a.dup})")

    # 학습 대상: 라벨이 있고 최소 1명이 푼 문제 (QASC 사다리와 같은 관례).
    # n_solved=0 문제도 이제는 정답코드가 있지만, 어떤 expert에도 배정되지 않으므로
    # 조건 간 비교를 위해 dense/MoE 모두 같은 문제집합만 본다.
    sft_rows, n_zero, n_missing = [], 0, 0
    label_rows = []
    for pid, per_expert in labels.items():
        r = by_pid.get(pid)
        if r is None:
            n_missing += 1
            continue
        n_solved = sum(per_expert.values())
        if n_solved == 0:
            n_zero += 1
            continue
        sft_rows.append(r)
        label_rows.append({"id": str(r["id"]), "dataset": "acc",
                           "n_solved": n_solved, "per_expert": per_expert})
    print(f"SFT 대상 {len(sft_rows)}문제 (전원 실패 제외 {n_zero}, 홀드아웃/누락 {n_missing})")

    out = corpus / "sft"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "acc_train.jsonl", "w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ("validation", "test"):
        src = corpus / f"acc_{name}.jsonl"
        if src.is_file():
            (out / f"acc_{name}.jsonl").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  -> {out}/acc_train.jsonl (+ validation/test 복사)")

    pkg = REPO / a.pkg
    pkg.mkdir(parents=True, exist_ok=True)
    with open(pkg / "binning_labels.jsonl", "w", encoding="utf-8") as f:
        for r in label_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mapping = {e: {"name": e, "system_prompt": "You are a helpful assistant.",
                   "strengths": f"acc {SEED} evolved expert", "train_pass_at_1": 0.0} for e in EX}
    json.dump(mapping, open(pkg / "agent_mapping.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"source_train_jsonl": str((out / "acc_train.jsonl").relative_to(REPO)),
               "experts": EX, "dup_rule": a.dup,
               "note": "SFT target = execution-verified reference solution from source dump"},
              open(pkg / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"라벨패키지: {len(label_rows)}문제 × {len(EX)} experts -> {pkg}")

    for cap in (10, 9, 8):
        v = {e: sum(1 for r in label_rows if r["per_expert"].get(e, 0) == 1 and r["n_solved"] <= cap)
             for e in EX}
        print(f"  cap{cap}: min={min(v.values())} max={max(v.values())} 평균={sum(v.values())//len(v)}")
    print(f"  shared(min_n_solved=10): {sum(1 for r in label_rows if r['n_solved'] >= 10)}")
    lens = sorted(len(r["solution"]) for r in sft_rows)
    com = [sum(1 for x in r["solution"].split("\n") if x.strip().startswith("#")) for r in sft_rows]
    print(f"  타깃 길이 p50={lens[len(lens)//2]} p90={lens[int(.9*len(lens))]}자, "
          f"주석 0줄인 비율={100*sum(1 for c in com if c == 0)//len(com)}%")


if __name__ == "__main__":
    main()

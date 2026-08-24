"""Super-NaturalInstructions(niv2) → export/sni/*.jsonl 빌더.

레포: https://github.com/allenai/natural-instructions (tasks/*.json + splits/{default,xlingual}/)

우리 파이프라인 계약(export/lbox와 동일):
  id / instruction / ground_truth / domain  + 로더가 dataset·scoring_kind를 덧붙인다.

⚠️ v2에서 바뀐 것 — **Definition을 instruction에 융합하지 않는다.**
v1은 `instruction = Definition + Input`이라 user 턴에 태스크 정의 537자가 들어갔고,
그게 조작과 출력형식을 다 못박아 페르소나(system 90자)가 개입할 자리를 없앴다.
프로브 job 229352의 null이 여기서 나왔다(docs/REFLECTION_sni_probe.md).

  instruction   : Input 원문만
  definition    : Definition 원문 — 프롬프트에 쓰지 않는다. 감사·판독용 보존.
  answer_line   : user 턴에 들어갈 답변공간 한 줄 (build_answer_line, 기계적 생성)
  answer_space  : 닫힌 태스크의 라벨 목록(원표기). 열린 태스크는 None.
  task_closed / n_gold_types / gold_len_median : 판독 층화 변수 (표집에는 안 쓴다)

SNI 고유 라벨(로스터 축):
  task_name / category / sni_domain / input_language / output_language

ground_truth는 **리스트**다. SNI는 한 인스턴스에 복수 정답을 허용하고 공식 채점도
레퍼런스 최대값을 쓰므로, 여기서 하나로 줄이면 안 된다(scorer가 max를 잡는다).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.scorer import _sni_normalize  # noqa: E402  (채점과 같은 정규화를 쓴다)


def _read_split(splits_dir: str, track: str, name: str) -> List[str]:
    path = os.path.join(splits_dir, track, f"{name}_tasks.txt")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _load_task(tasks_dir: str, task_name: str) -> Dict[str, Any] | None:
    path = os.path.join(tasks_dir, f"{task_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _first(v: Any, default: str = "unknown") -> str:
    if isinstance(v, list):
        return str(v[0]) if v else default
    return str(v or default)


# --- answer_line: user 턴에 들어갈 유일한 지시 -------------------------------
# 규칙은 기계적이다(LLM 미사용). 이유는 두 가지 —
#  (1) 조작("무엇을 하라")이 새어 들어가면 안 된다. 그건 페르소나가 담아야 하고,
#      그게 이 프로브의 관측 대상이다.
#  (2) 875 task 전수를 사람이 감사할 수 있어야 한다.
MAX_LABELS = 20          # 라벨이 이보다 많으면 나열하지 않고 열린 태스크로 본다
CLOSED_RATIO = 0.2       # 정답 종수 / 인스턴스 수. 이보다 크면 열린 태스크
MAX_LABEL_TOKENS = 8     # 라벨 하나가 이보다 길면 '라벨'이 아니다 → 열린 태스크
OPEN_LINE = "Give the output only, with no explanation."

# MAX_LABEL_TOKENS의 근거(v2 빌드 전수 실측): 닫힘 443종 중 388종이 1토큰, 427종이 2토큰 이하고,
# 최대 라벨 길이는 5토큰 다음 11 → 16 → 51로 건너뛴다. 그 위쪽 3종
# (task599_cuad_question_generation 51토큰 등)은 전부 생성형 태스크가 정답 반복 때문에
# 닫힘으로 잘못 잡힌 것이고, 나열하면 user 턴이 4,586자가 되어
# 원 프로브에서 페르소나를 덮었던 '거대한 user 턴'을 그대로 재현한다.


def task_answer_profile(
    instances: List[Dict[str, Any]],
) -> Tuple[bool, List[str], int, float]:
    """(닫힘 여부, 라벨 목록(원표기), 정답 종수, gold 토큰길이 중앙값).

    닫힘 판정은 데이터만 본다 — 태스크 안에서 정답이 소수의 값으로 반복되면 분류형이다.
    Definition을 읽지 않으므로 판정에 설계자의 해석이 들어가지 않는다.
    """
    surface: Dict[str, Counter] = {}
    lens: List[int] = []
    for inst in instances:
        outs = inst.get("output") or []
        if not isinstance(outs, list):
            outs = [outs]
        first = True
        for g in outs:
            g = str(g)
            n = _sni_normalize(g)
            if not n:
                continue
            surface.setdefault(n, Counter())[g.strip()] += 1
            if first:
                lens.append(len(n.split()))
                first = False
    n_types = len(surface)
    if not instances or not n_types:
        return False, [], n_types, 0.0
    closed = n_types <= MAX_LABELS and (n_types / len(instances)) <= CLOSED_RATIO
    labels: List[str] = []
    if closed:
        # 같은 라벨의 표기가 여럿이면 가장 흔한 표기를 대표로 쓴다.
        labels = sorted(c.most_common(1)[0][0] for c in surface.values())
        if any(len(x.split()) > MAX_LABEL_TOKENS for x in labels):
            closed, labels = False, []
    return closed, labels, n_types, (statistics.median(lens) if lens else 0.0)


def build_answer_line(closed: bool, labels: List[str], out_lang: str) -> str:
    """user 턴 한 줄. **열린 태스크에 길이·형식을 규정하지 않는다** —
    길이를 지정하면 이 실험이 재려는 문체를 프롬프트가 먼저 죽인다."""
    line = f"Answer with exactly one of: {', '.join(labels)}." if closed else OPEN_LINE
    if out_lang and out_lang not in ("English", "unknown"):
        line = f"{line} Answer in {out_lang}."
    return line


def build(
    repo: str,
    out_dir: str,
    track: str,
    per_task: int,
    seed: int,
    english_only: bool,
) -> None:
    tasks_dir = os.path.join(repo, "tasks")
    splits_dir = os.path.join(repo, "splits")
    os.makedirs(out_dir, exist_ok=True)

    stats: Dict[str, Dict[str, int]] = {}
    all_rows: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for split in ("train", "test"):
        names = _read_split(splits_dir, track, split)
        rows: List[Dict[str, Any]] = []
        n_tasks = 0
        skipped_lang = 0
        for task_name in names:
            task = _load_task(tasks_dir, task_name)
            if task is None:
                continue
            in_lang = _first(task.get("Input_language"))
            out_lang = _first(task.get("Output_language"))
            # Domains는 계층 라벨이다("Social Media -> Twitter"). 최상위만 축으로 쓴다 —
            # 부모와 자식을 별도 전문가로 두는 건 의미가 없다. 원본도 같이 남긴다.
            sni_domain_full = _first(task.get("Domains"))
            sni_domain = sni_domain_full.split("->")[0].strip()
            if english_only and not (in_lang == "English" and out_lang == "English"):
                skipped_lang += 1
                continue
            definition = _first(task.get("Definition"), "")
            category = _first(task.get("Categories"))
            # 공식 표준 프롬프트는 정의 + positive example 2건이다
            # (Tk-Instruct scripts/{train,eval}_tk_instruct.sh: --num_pos_examples 2).
            # v2까지 이걸 버리고 zero-shot으로 갔던 건 근거가 틀린 자체 판단이었다.
            pos_ex = [
                {"input": str(e.get("input") or "").strip(),
                 "output": str(e.get("output") or "").strip(),
                 "explanation": str(e.get("explanation") or "").strip()}
                for e in (task.get("Positive Examples") or [])
            ]
            neg_ex = [
                {"input": str(e.get("input") or "").strip(),
                 "output": str(e.get("output") or "").strip(),
                 "explanation": str(e.get("explanation") or "").strip()}
                for e in (task.get("Negative Examples") or [])
            ]
            instances = task.get("Instances") or []
            rng = random.Random(f"{seed}:{task_name}")
            if len(instances) > per_task:
                instances = rng.sample(instances, per_task)
            n_tasks += 1
            # answer_line은 태스크 단위로 한 번 결정된다(표집된 인스턴스만 보고).
            closed, labels, n_types, gold_len = task_answer_profile(instances)
            answer_line = build_answer_line(closed, labels, out_lang)
            audit.append(
                {
                    "task_name": task_name, "split": split, "category": category,
                    "closed": closed, "n_types": n_types, "n_inst": len(instances),
                    "gold_len_median": gold_len, "answer_line": answer_line,
                    "definition": definition,
                }
            )
            for inst in instances:
                gts = inst.get("output") or []
                if not isinstance(gts, list):
                    gts = [gts]
                gts = [str(g) for g in gts if str(g).strip()]
                if not gts:
                    continue
                rows.append(
                    {
                        "id": str(inst.get("id") or f"{task_name}-{len(rows)}"),
                        # ⚠️ Input 원문만. Definition은 아래 별도 필드로만 남는다.
                        "instruction": str(inst.get("input") or "").strip(),
                        "ground_truth": gts,
                        "domain": "sni",
                        "definition": definition,
                        "positive_examples": pos_ex,
                        "negative_examples": neg_ex,
                        "answer_line": answer_line,
                        "answer_space": labels or None,
                        "task_closed": closed,
                        "n_gold_types": n_types,
                        "gold_len_median": gold_len,
                        "task_name": task_name,
                        "category": category,
                        "sni_domain": sni_domain,
                        "sni_domain_full": sni_domain_full,
                        "input_language": in_lang,
                        "output_language": out_lang,
                    }
                )
        path = os.path.join(out_dir, f"sni_{split}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {
            "tasks": n_tasks,
            "instances": len(rows),
            "skipped_non_english_tasks": skipped_lang,
        }
        print(f"[{split}] tasks={n_tasks} instances={len(rows)} -> {path}")
        all_rows.extend(rows)

    # 공식 split은 train/test의 category가 완전 disjoint(cross-task generalization용)라
    # 우리 용도에 못 쓴다. 합친 한 덩어리를 대상 모집단으로 삼는다.
    path_all = os.path.join(out_dir, "sni_all.jsonl")
    with open(path_all, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[all] instances={len(all_rows)} -> {path_all}")

    n_closed = sum(1 for a in audit if a["closed"])
    with open(os.path.join(out_dir, "build_report.json"), "w") as f:
        json.dump(
            {
                "repo": repo,
                "track": track,
                "per_task": per_task,
                "seed": seed,
                "english_only": english_only,
                "splits": stats,
                "all_instances": len(all_rows),
                "answer_line_rule": {
                    "max_labels": MAX_LABELS,
                    "closed_ratio": CLOSED_RATIO,
                    "open_line": OPEN_LINE,
                    "closed_tasks": n_closed,
                    "open_tasks": len(audit) - n_closed,
                },
            },
            f,
            indent=2,
        )
    return audit


def write_audit(audit: List[Dict[str, Any]], path: str) -> None:
    """875 task 전수의 answer_line을 사람이 검수할 수 있게 덤프한다.
    검수 포인트: **조작("무엇을 하라")이 새어 들어간 줄이 있는가.**"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_closed = sum(1 for a in audit if a["closed"])
    L = [
        "# answer_line 전수 감사 — user 턴에 들어갈 유일한 지시",
        "",
        f"- task {len(audit)}종 · 닫힘 {n_closed} / 열림 {len(audit) - n_closed}",
        f"- 규칙: 정답 종수 ≤ {MAX_LABELS} **그리고** 종수/인스턴스 ≤ {CLOSED_RATIO} → 닫힘(라벨 나열),",
        f"  그 외 → `{OPEN_LINE}` · 출력언어가 English가 아니면 `Answer in {{language}}.` 덧붙임",
        "- **검수 포인트: 조작(무엇을 하라)이 새어 들어간 줄이 있는가.** Definition은 비교용으로만 싣는다.",
        "",
        "| task | cat | 닫힘 | 종수/인스턴스 | gold중앙 | answer_line |",
        "|---|---|---|---:|---:|---|",
    ]
    for a in sorted(audit, key=lambda x: x["task_name"]):
        line = a["answer_line"].replace("|", "\\|")
        if len(line) > 160:
            line = line[:157] + "..."
        L.append(
            f"| {a['task_name'][:46]} | {a['category'][:20]} | {'C' if a['closed'] else 'O'} "
            f"| {a['n_types']}/{a['n_inst']} | {a['gold_len_median']:.0f} | {line} |"
        )
    L += ["", "---", "", "## Definition 대조(열린 태스크 표본 20종)", ""]
    for a in [x for x in audit if not x["closed"]][:20]:
        L += [f"### {a['task_name']}", "", "```", a["definition"][:600], "```",
              f"→ answer_line: `{a['answer_line']}`", ""]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[audit] {len(audit)} tasks -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/data5/jaehoonjeong/datasets/natural-instructions")
    ap.add_argument("--out", default="export/sni")
    ap.add_argument("--track", default="default", choices=["default", "xlingual"])
    ap.add_argument("--per-task", type=int, default=100, help="태스크당 인스턴스 상한(공식 관례 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--english-only",
        action="store_true",
        help="Input/Output이 모두 English인 태스크만. arm A/B용. arm C(언어 축)는 끄고 쓴다.",
    )
    ap.add_argument("--audit-out", default="results/sni/answer_lines.md",
                    help="answer_line 전수 감사 덤프. 승인 전에는 이 export를 런에 쓰지 않는다.")
    args = ap.parse_args()
    audit = build(args.repo, args.out, args.track, args.per_task, args.seed, args.english_only)
    write_audit(audit, args.audit_out)


if __name__ == "__main__":
    main()

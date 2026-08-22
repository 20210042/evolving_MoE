"""Super-NaturalInstructions(niv2) → export/sni/*.jsonl 빌더.

레포: https://github.com/allenai/natural-instructions (tasks/*.json + splits/{default,xlingual}/)

우리 파이프라인 계약(export/lbox와 동일):
  id / instruction / ground_truth / domain  + 로더가 dataset·scoring_kind를 덧붙인다.

SNI 고유 필드도 함께 싣는다 — 프로브 3-arm이 전부 이 메타데이터를 축으로 쓴다.
  task_name        : task001_quoref_question_generation  (태스크 단위 split 확인용)
  category         : Categories[0]  → arm B(태스크유형 축)
  input_language / output_language  → arm C(언어 축, positive control)

ground_truth는 **리스트**다. SNI는 한 인스턴스에 복수 정답을 허용하고 공식 채점도
레퍼런스 최대값을 쓰므로, 여기서 하나로 줄이면 안 된다(scorer가 max를 잡는다).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List


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


def build_instruction(definition: str, inp: str) -> str:
    """Definition(태스크 지시문) + Input. few-shot은 붙이지 않는다 —
    baseline은 zero-shot definition-only가 SNI 공식 기본 설정이고, few-shot을 섞으면
    페르소나 효과가 예제에 묻혀 프로브의 관측 대상이 흐려진다."""
    return f"{definition.strip()}\n\nInput:\n{inp.strip()}"


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
            instances = task.get("Instances") or []
            rng = random.Random(f"{seed}:{task_name}")
            if len(instances) > per_task:
                instances = rng.sample(instances, per_task)
            n_tasks += 1
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
                        "instruction": build_instruction(definition, str(inst.get("input") or "")),
                        "ground_truth": gts,
                        "domain": "sni",
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

    with open(os.path.join(out_dir, "build_report.json"), "w") as f:
        json.dump(
            {
                "repo": repo,
                "track": track,
                "per_task": per_task,
                "seed": seed,
                "english_only": english_only,
                "splits": stats,
            },
            f,
            indent=2,
        )


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
    args = ap.parse_args()
    build(args.repo, args.out, args.track, args.per_task, args.seed, args.english_only)


if __name__ == "__main__":
    main()

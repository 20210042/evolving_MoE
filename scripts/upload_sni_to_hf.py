#!/usr/bin/env python3
"""export/sni_v4 를 HuggingFace Dataset 레포로 올린다 (협업자 공유용).

⚠️ 업로드는 되돌리기 어려운 외부 공개 동작이다. --yes 없이는 올리지 않고
   무엇을 올릴지만 출력한다(dry-run이 기본).

원본: allenai/natural-instructions (Apache-2.0). 재배포 가능하되 출처를 남긴다.

Usage:
  python scripts/upload_sni_to_hf.py --repo-id <org>/<name> --private          # dry-run
  python scripts/upload_sni_to_hf.py --repo-id <org>/<name> --private --yes    # 실제 업로드
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data5/jaehoonjeong/.cache/huggingface")

CARD = """---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- super-natural-instructions
- niv2
- mixture-of-experts
---

# SNI export ({version})

`allenai/natural-instructions` (Super-NaturalInstructions / niv2, Apache-2.0)를
우리 파이프라인 계약에 맞춰 재가공한 것이다. **원본 태스크·정답은 그대로이고,
바뀐 것은 필드 구성과 분할이다.**

## 무엇이 들어있나

| 파일 | 인스턴스 |
|---|---:|
{table}

- `sni_train / sni_valid / sni_test` — **인스턴스 단위 무작위 {ratio} 분할** (seed {seed}).
  같은 태스크의 인스턴스가 분할을 가로질러 흩어진다.
{extra_files}

## 필드

| 필드 | 뜻 |
|---|---|
| `id` | 원본 인스턴스 id |
| `instruction` | Input **원문만** (Definition을 융합하지 않는다) |
| `ground_truth` | 정답 **리스트** — SNI는 복수 정답을 허용하고 공식 채점도 레퍼런스 최대값을 쓴다 |
| `definition` | 태스크 정의 원문 |
| `positive_examples` / `negative_examples` | 원본 예시. 공식 표준은 pos 2건 / neg 0건 |
| `task_name` / `category` / `sni_domain` | 로스터 축 라벨 |
| `task_closed` / `answer_space` / `n_gold_types` / `gold_len_median` | 판독 층화 변수 |
| `input_language` / `output_language` | 언어 |

## 프롬프트 조립 (공식 Tk-Instruct 형식)

```
system: <페르소나>\\n\\n<definition>
user:    Positive Example 1 - / Input: ... / Output: ...
         Positive Example 2 - / Input: ... / Output: ...
         Now complete the following example -
         Input: <instruction>
         Output:
```

공식 표준값은 `--num_pos_examples 2 --num_neg_examples 0 --add_explanation False
--max_target_length 128` (`yizhongw/Tk-Instruct`의 `scripts/{{train,eval,gpt3}}_tk_instruct.sh`).

## 채점

공식 `eval/automatic/evaluation.py` 그대로 — 태스크 분기 없이 **EM과 ROUGE-L을 병기**한다
(`rouge_score`, `use_stemmer=True`). 우리 이식본은 실제 예측 203,972건 대조에서 불일치 0건.

## 제외분

프롬프트가 `max_model_len`(16,384)을 넘는 인스턴스 {n_excluded}건을 뺐다. 범위 판단이 아니라
모델 한계다(초과 배치가 들어오면 vllm이 잡을 죽인다). 목록은 `excluded_over_context.json`.

## 출처

원본: https://github.com/allenai/natural-instructions (Apache-2.0)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="export/sni_v4")
    ap.add_argument("--repo-id", required=True, help="예: myorg/sni-moe-v4")
    ap.add_argument("--private", action="store_true", help="비공개 레포로 만든다(권장)")
    ap.add_argument("--excluded", default="results/sni/excluded_over_context_v4.json")
    ap.add_argument("--slim", action="store_true",
                    help="sni_{train,valid,test} + 메타데이터만 올린다. sni_all과 official_*은 "
                         "같은 인스턴스의 중복본이라 뺀다(셋으로 복원 가능).")
    ap.add_argument("--no-card", action="store_true",
                    help="데이터셋 카드를 만들지 않는다. 이미 올라간 README.md도 건드리지 않는다.")
    ap.add_argument("--yes", action="store_true", help="실제로 업로드한다. 없으면 dry-run.")
    a = ap.parse_args()

    SLIM_KEEP = {"sni_train.jsonl", "sni_valid.jsonl", "sni_test.jsonl",
                 "README.md", "build_report.json", "excluded_over_context.json"}

    src = Path(a.src)
    files = sorted(p for p in src.glob("*.jsonl")) + sorted(src.glob("*.json"))
    if not files:
        raise SystemExit(f"{src}에 올릴 파일이 없다 — 먼저 재빌드를 돌려라.")

    report = json.loads((src / "build_report.json").read_text(encoding="utf-8"))
    counts = {p.stem: sum(1 for _ in p.open()) for p in src.glob("*.jsonl")
              if not (a.slim and p.name not in SLIM_KEEP)}
    table = "\n".join(f"| `{k}` | {v:,} |" for k, v in sorted(counts.items()))
    ratio = ":".join(str(x) for x in (report.get("split_ratio") or []))
    n_excluded = report.get("excluded_over_context", 0)

    extra = ("- 공식 category-disjoint split(`official_*`)과 전수 합본(`sni_all`)은 위 셋으로\n"
             "  복원 가능해 올리지 않았다. 공식 split 멤버십은 원본 레포의\n"
             "  `splits/default/*_tasks.txt` + 각 행의 `task_name`으로 재구성한다."
             if a.slim else
             "- `official_train / official_test` — 공식 split. category가 완전 disjoint\n"
             "  (cross-task generalization용)라 우리 용도엔 맞지 않아 **참고용으로만** 남긴다.\n"
             "- `sni_all` — 위 셋을 합친 전수.")
    if a.no_card:
        card_path = src / "README.md"
        print(f"[--no-card] 카드를 만들지 않는다. 현재 {card_path}: "
              f"{card_path.stat().st_size if card_path.exists() else '없음'} bytes")
        _skip_card = True
    else:
        _skip_card = False
    card = None if a.no_card else CARD.format(version=src.name, table=table, ratio=ratio or "미분할",
                       seed=report.get("seed"), n_excluded=n_excluded, extra_files=extra)
    if not _skip_card:
        card_path = src / "README.md"
        card_path.write_text(card, encoding="utf-8")

    excl = Path(a.excluded)
    if excl.exists():
        (src / "excluded_over_context.json").write_text(
            excl.read_text(encoding="utf-8"), encoding="utf-8")

    sending = [p for p in sorted(src.iterdir())
               if p.is_file() and not (a.slim and p.name not in SLIM_KEEP)]
    total = sum(p.stat().st_size for p in sending)
    print(f"대상: {src} -> {a.repo_id} ({'private' if a.private else 'PUBLIC'})"
          f"{' · slim' if a.slim else ''}")
    for p in sending:
        print(f"  {p.name:32s} {p.stat().st_size/1e6:9.1f} MB")
    if a.slim:
        for p in sorted(src.iterdir()):
            if p.is_file() and p.name not in SLIM_KEEP:
                print(f"  (제외) {p.name}")
    print(f"  합계 {total/1e6:.1f} MB")
    if not _skip_card:
        print(f"\n데이터셋 카드 -> {card_path}")

    if not a.yes:
        print("\n[dry-run] --yes 없이는 업로드하지 않는다. 카드를 검토한 뒤 --yes로 다시 실행해라.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(src), repo_id=a.repo_id, repo_type="dataset",
                      allow_patterns=sorted(SLIM_KEEP) if a.slim else None)
    print(f"\n업로드 완료 -> https://huggingface.co/datasets/{a.repo_id}")


if __name__ == "__main__":
    main()

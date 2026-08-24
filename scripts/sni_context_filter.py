#!/usr/bin/env python3
"""컨텍스트 한계를 넘는 SNI 아이템 id 목록을 뽑는다 (사전등록된 기술적 제외).

범위 판단이 아니라 모델 한계다 — 프롬프트가 max_model_len을 넘으면 vllm이 배치 전체를
거부하고 잡이 죽는다(job 229520이 그렇게 실패했다). 제외분은 목록으로 남겨 감사 가능하게 한다.

프롬프트는 실제 경로와 동일하게 조립한다:
  system = 페르소나 + definition   /  user = answer_line + instruction
페르소나는 로스터 중 **가장 긴 것**을 써서 보수적으로 잰다(어느 expert에서도 안 넘게).

Usage:
  python scripts/sni_context_filter.py --data export/sni_v2/sni_all.jsonl \
      --roster configs/roster_sni_probe_v2.json --limit 16384 \
      --out results/sni/excluded_over_context.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data5/jaehoonjeong/.cache/huggingface")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="export/sni_v2/sni_all.jsonl")
    ap.add_argument("--roster", default="configs/roster_sni_probe_v2.json")
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--limit", type=int, default=16384, help="max_model_len")
    ap.add_argument("--out", default="results/sni/excluded_over_context.json")
    ap.add_argument("--screen_chars", type=int, default=8000,
                    help="이 문자수 이하는 토큰화하지 않는다(안전 마진: 8k자면 <3k토큰)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model)

    roster = json.loads(Path(a.roster).read_text(encoding="utf-8"))
    persona = max((p.get("system_prompt", "") for p in roster), key=len)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    over, maxtok = [], 0
    for r in rows:
        approx = (len(persona) + len(r.get("definition") or "")
                  + len(r.get("answer_line") or "") + len(r["instruction"]))
        if approx <= a.screen_chars:
            continue
        sys_t = f"{persona}\n\n{(r.get('definition') or '').strip()}".strip()
        usr = f"{(r.get('answer_line') or '').strip()}\n\n{r['instruction']}".strip()
        txt = tk.apply_chat_template(
            [{"role": "system", "content": sys_t}, {"role": "user", "content": usr}],
            tokenize=False, add_generation_prompt=True)
        # ⚠️ apply_chat_template(tokenize=True)는 BatchEncoding을 돌려줘 len()이 키 개수(2)가 된다.
        n = len(tk(txt, add_special_tokens=False)["input_ids"])
        maxtok = max(maxtok, n)
        if n > a.limit:
            over.append({"id": r["id"], "task_name": r["task_name"], "tokens": n})

    by_task = collections.Counter(o["task_name"] for o in over)
    out = {
        "limit": a.limit, "model": a.model, "persona_used": persona,
        "total_items": len(rows), "excluded": len(over),
        "max_tokens_seen": maxtok, "by_task": dict(by_task),
        "ids": [o["id"] for o in over],
        "detail": sorted(over, key=lambda x: -x["tokens"]),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"제외 {len(over)}건 / {len(rows):,} ({100*len(over)/len(rows):.3f}%) "
          f"· 최대 {maxtok:,}토큰 -> {a.out}")
    for k, v in by_task.most_common():
        print(f"   {k}: {v}건")


if __name__ == "__main__":
    main()

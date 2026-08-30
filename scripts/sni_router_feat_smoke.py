#!/usr/bin/env python
"""라우터 특징 추출 스모크 — 본 잡 전에 두 가지를 실측한다.

(1) 토큰 길이 분포: SNI 프롬프트를 실제 조립기(sni_system_block/sni_user_block)로 만들고
    전수(test 8,699 + train 69,588)를 토크나이즈해 p50/p90/p95/p99/max를 낸다.
    → max_len을 추측하지 않고 고른다. HF truncation은 뒤를 자르므로 짧게 잡으면 Input이 날아간다.
(2) forward 처리량: max_len 후보별로 소수 문제를 실제로 돌려 초/문제를 재고 78,287문제 ETA를 낸다.
    A안(문제당 1회)과 B안(문제×16명)을 같은 측정에서 환산한다.

Usage: python3 scripts/sni_router_feat_smoke.py --n 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from prompts import baseline_prompts as bp            # noqa: E402
from prompts.coding import sni_system_block, sni_user_block  # noqa: E402

MODEL = "google/gemma-4-26B-A4B-it"
ROSTER = REPO / "results/sni/seed20212003/roster_final.json"


def build(row, persona=None):
    """페르소나 자리를 비우면(=중립) A안 문제 표현, 페르소나를 넣으면 B안 쌍 표현."""
    sys_p = sni_system_block(persona or bp.SNI_GEN_SYSTEM, row.get("definition"))
    usr = sni_user_block(row.get("answer_line"), row["instruction"],
                         positive_examples=row.get("positive_examples"), num_pos=2)
    return sys_p, usr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_lens", type=int, nargs="*", default=[1024, 2048, 4096])
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def render(row, persona=None):
        s, u = build(row, persona)
        return tok.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True)

    # ---- (1) 토큰 길이 전수
    print("=== 토큰 길이 (실제 조립: 정의 + 예시2 + Input, 페르소나 중립)", flush=True)
    stats = {}
    for split, path in (("test", "export/sni_v4/sni_test.jsonl"),
                        ("train", "export/sni_v4/sni_train.jsonl")):
        rows = [json.loads(l) for l in open(REPO / path, encoding="utf-8")]
        L = np.array([len(tok(render(r), add_special_tokens=False)["input_ids"])
                      for r in rows])
        stats[split] = L
        q = np.percentile(L, [50, 90, 95, 99])
        print(f"  {split:5s} n={len(L):,}  p50 {q[0]:.0f} · p90 {q[1]:.0f} · "
              f"p95 {q[2]:.0f} · p99 {q[3]:.0f} · max {L.max()}", flush=True)
        for m in a.max_lens:
            print(f"        max_len={m:5d} → 잘리는 문제 {(L > m).mean()*100:5.2f}% "
                  f"({int((L > m).sum()):,}건)", flush=True)

    # 페르소나를 넣으면 얼마나 길어지는가(B안)
    roster = json.load(open(ROSTER, encoding="utf-8"))
    rows = [json.loads(l) for l in open(REPO / "export/sni_v4/sni_test.jsonl",
                                        encoding="utf-8")][:a.n]
    add = [len(tok(render(r, p["system_prompt"]), add_special_tokens=False)["input_ids"])
           - len(tok(render(r), add_special_tokens=False)["input_ids"])
           for r in rows[:50] for p in roster[:3]]
    print(f"  페르소나 삽입 시 증가 토큰: 중앙값 {int(np.median(add))} "
          f"(min {min(add)}, max {max(add)})", flush=True)

    # ---- (2) forward 처리량
    print("\n=== forward 처리량", flush=True)
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True).cuda().eval()
    print(f"  모델 로드 {time.time()-t0:.0f}s · "
          f"GPU mem {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)

    N_TOTAL = 69588 + 8699
    for m in a.max_lens:
        texts = [render(r) for r in rows]
        t0 = time.time()
        for i in range(0, len(texts), a.batch):
            enc = tok(texts[i:i + a.batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=m).to("cuda")
            with torch.no_grad():
                model(**enc, output_hidden_states=True)
        dt = time.time() - t0
        per = dt / len(texts)
        print(f"  max_len={m:5d}  {per*1000:6.1f} ms/문제 → "
              f"A안(문제당 1회) {per*N_TOTAL/3600:5.2f}h · "
              f"B안(×16명) {per*N_TOTAL*16/3600:6.2f}h · "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""base 모델의 '답 레터 분포' 추출 — 라우터용 task-aware 특징.

프롬프트(baseline GEN, add_generation_prompt) 다음 토큰 위치에서 정답 레터 토큰의
로짓을 뽑아 softmax → (N, |레터|) 답 분포. base의 confidence/난이도 신호 = solvability와 상관.

⚠️ **MCQA 전용.** 고정된 답 레터 어휘(QASC의 A~H)를 전제한다. LBOX처럼 답을 생성해
문자열로 채점하는 오픈 QA에는 이 정의가 성립하지 않아 require_mcqa()가 막는다.

결과: results/embed_viz_test/<ds>_<split>_ansprob.npy + _ids.json
Usage: python scripts/extract_answer_logits.py --dataset qasc --split train
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

REPO = rc.REPO
OUT = rc.FEAT_DIR


def main():
    ap = argparse.ArgumentParser()
    rc.add_dataset_arg(ap)
    ap.add_argument("--split", default=None, help="기본값은 데이터셋의 train split")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=1024)
    a = ap.parse_args()
    sp = rc.spec(a.dataset)
    split = a.split or sp.train_split
    # 답 레터 분포는 MCQA 전용 — 오픈 QA면 여기서 막힌다.
    rc.require_mcqa(sp, "answer-logits(답 레터 분포) 특징")
    LETTERS = list(sp.answer_letters)
    sys_p, user_t = sp.gen
    MODEL = sp.base_model
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(REPO / sp.src[split], encoding="utf-8")]
    ids = [str(r["id"]) for r in rows]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()

    # 각 레터의 토큰 id (공백 포함/미포함 변형 중 유효한 것 모아서 로짓 max)
    letter_ids = []
    for ch in LETTERS:
        cand = set()
        for s in (ch, " " + ch):
            t = tok.encode(s, add_special_tokens=False)
            if t:
                cand.add(t[0])
        letter_ids.append(sorted(cand))

    def prompt(r):
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": user_t.format(instruction=r["instruction"])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    probs = []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        enc = tok([prompt(r) for r in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=a.max_len).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()   # (B,V) 다음 토큰
        letter_logit = torch.stack(
            [logits[:, ids_].max(1).values for ids_ in letter_ids], dim=1)  # (B,8)
        probs.append(torch.softmax(letter_logit, dim=1).cpu().numpy())
        if (i // a.batch) % 20 == 0:
            print(f"{i+len(chunk)}/{len(rows)}", flush=True)

    P = np.concatenate(probs).astype(np.float32)
    np.save(rc.feat_path(sp, split, "ansprob"), P)
    json.dump(ids, open(rc.feat_path(sp, split, "ansprob_ids"), "w"))
    print(f"done -> {rc.feat_path(sp, split, 'ansprob').name} {P.shape}")


if __name__ == "__main__":
    main()

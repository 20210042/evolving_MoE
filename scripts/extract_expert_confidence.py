#!/usr/bin/env python3
"""Expert 자기확신 라우팅 (near-oracle 후보 아키텍처).

각 학습된 어댑터를 끼우고 답 레터 분포를 뽑아, 그 expert가 자기 답에 얼마나
확신하는지(max prob)를 계산. 라우팅 = 문제별로 가장 확신하는 expert 선택.
confidence가 정답과 calibrated면 top-1이 oracle에 근접.

입력이 아니라 expert-side 계산을 쓰는 게 핵심 — solvability⊥input 한계를 우회.

⚠️ **MCQA 전용.** 확신도를 "다음 토큰 위치에서 정답 레터(A~H) softmax의 최대값"으로
정의하기 때문에 고정된 답 어휘와 단일 토큰 답을 전제한다. LBOX처럼 답을 생성해
문자열 EM으로 채점하는 오픈 QA에서는 이 정의가 성립하지 않으므로, 그 도메인에
쓰려면 확신도 정의부터 새로 세워야 한다(require_mcqa()가 막아둔다).

결과: results/embed_viz_test/<ds>_<split>_expert_conf.npz (conf, pred, ids, experts)
Usage: python scripts/extract_expert_confidence.py --dataset qasc --split validation
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
    ap.add_argument("--split", default=None, help="기본값은 데이터셋의 eval split")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--ckpt", default=None, help="어댑터 디렉터리 override")
    a = ap.parse_args()
    sp = rc.spec(a.dataset)
    split = a.split or sp.eval_split
    rc.require_mcqa(sp, "expert confidence(자기확신) 특징")
    LETTERS = list(sp.answer_letters)
    sys_p, user_t = sp.gen
    BASE = sp.base_model
    CKPT = Path(a.ckpt) if a.ckpt else REPO / sp.ckpt
    OUT.mkdir(parents=True, exist_ok=True)

    experts = sorted([p.name for p in CKPT.iterdir()
                      if p.is_dir() and (p / "adapter_config.json").is_file()])
    print("experts:", experts, flush=True)
    rows = [json.loads(l) for l in open(REPO / sp.src[split], encoding="utf-8")]
    ids = [str(r["id"]) for r in rows]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()
    # 첫 어댑터로 PeftModel 만들고 나머지는 load_adapter로 추가(베이스 1회 로드)
    model = PeftModel.from_pretrained(base, str(CKPT / experts[0]), adapter_name=experts[0])
    for e in experts[1:]:
        model.load_adapter(str(CKPT / e), adapter_name=e)
    model.eval()

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

    texts = [prompt(r) for r in rows]
    conf = np.zeros((len(rows), len(experts)), np.float32)   # max prob (자기확신)
    pred = np.zeros((len(rows), len(experts)), np.int64)      # argmax letter idx
    n_letters = len(LETTERS)
    for ei, e in enumerate(experts):
        model.set_adapter(e)
        for i in range(0, len(rows), a.batch):
            enc = tok(texts[i:i + a.batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len).to("cuda")
            with torch.no_grad():
                lg = model(**enc).logits[:, -1, :].float()
            ll = torch.stack([lg[:, ids_].max(1).values for ids_ in letter_ids], dim=1)  # (B,8)
            pr = torch.softmax(ll, dim=1)
            conf[i:i + a.batch, ei] = pr.max(1).values.cpu().numpy()
            pred[i:i + a.batch, ei] = pr.argmax(1).cpu().numpy()
        print(f"  {e} done", flush=True)

    out = rc.feat_path(sp, split, "conf")
    np.savez(out, conf=conf, pred=pred, ids=np.array(ids), experts=np.array(experts))
    print(f"done -> {out.name} {conf.shape}")


if __name__ == "__main__":
    main()

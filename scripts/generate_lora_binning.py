#!/usr/bin/env python3
"""Phase 1: 우리 아키텍처를 llama backbone에서 실측.

per-expert LoRA 어댑터를 base에 붙여 각 expert가 홀드아웃을 '실제 생성'하고,
score_binning으로 채점 가능한 형태로 떨군다.

MCQA(qasc)는 같은 generate 호출에서 첫 스텝 answer-logit(A-H)도 잡아 real
confidence를 동시 산출한다. 생성형 도메인(acc 등)은 답 글자가 없어 conf를 만들지
않는다 — output_scores를 켜면 (스텝×배치×어휘) 텐서가 수십 GB로 터지기도 한다.

프롬프트는 build_baseline_prompt를 그대로 탄다 = train_sft.py가 학습에 쓴 것과
동일 경로. 학습/평가 프롬프트가 갈리면 수치가 통째로 무의미해진다.

- 출력① inference_<split>_<tag>.jsonl : {id,dataset,ground_truth,scoring_kind,instruction,
        expert_outputs:{expert: 생성텍스트}}  (score_binning으로 채점)
- 출력② <ds>_<split>_<tag>_conf.npz : conf/pred (N×E, real) — MCQA 전용

Usage:
  python scripts/generate_lora_binning.py --split validation                      # qasc(기본)
  python scripts/generate_lora_binning.py --dataset acc --tag evolved \
      --src export/acc/acc_eval_clean290.jsonl \
      --ckpt checkpoints/expert_sft/acc_seed20210111_cap9 \
      --max_new_tokens 2048 --max_len 4096 --batch 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
sys.path.insert(0, str(REPO / "src"))
from prompts.coding import build_baseline_prompt  # noqa: E402

OUTNPZ = REPO / "results" / "embed_viz_test"
BASE = "meta-llama/Llama-3.1-8B-Instruct"

# 도메인 기본값. --src/--out_dir/--ckpt로 덮어쓴다.
DATASETS = {
    "qasc": dict(
        src={"train": "export/qasc/qasc_train.jsonl",
             "validation": "export/qasc/qasc_validation.jsonl"},
        res="results/qasc/seed20210211",
        ckpt="checkpoints/expert_sft/qasc_seed20210211_cap10",
        letters=list("ABCDEFGH"),
    ),
    "acc": dict(
        src={"train": "export/acc/acc_train.jsonl",
             "validation": "export/acc/acc_validation.jsonl"},
        res="results/acc/seed20210111",
        ckpt="checkpoints/expert_sft/acc_seed20210111_cap9",
        letters=None,          # 코드 생성 — 답 글자가 없다
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="qasc", choices=sorted(DATASETS))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--src", default=None, help="입력 jsonl override (홀드아웃 부분집합 등)")
    ap.add_argument("--out_dir", default=None, help="결과 디렉터리 override")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="스모크용: 앞 N문제만")
    ap.add_argument("--ckpt", default=None, help="어댑터 로스터 디렉토리")
    ap.add_argument("--tag", default="lora13", help="출력 파일 태그 (inference_<split>_<tag>.jsonl)")
    # greedy는 반복 루프에서 빠져나올 확률적 탈출구가 없다. 코딩 도메인에서 실제로
    # 생성의 41%가 주석을 무한반복하며 토큰 상한까지 갔다(중복라인 0.68). 진화 경로가
    # 쓰던 1.05를 그대로 쓴다. 1.0 = 끄기 = 기존 동작.
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    # 전문가 단위 부분 저장은 항상 한다. --resume이면 완결된 전문가는 건너뛴다
    # (28시간짜리 잡이 마지막에 죽어 전부 날아가는 걸 막는다).
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    ds = DATASETS[a.dataset]
    LETTERS = ds["letters"]
    MCQA = LETTERS is not None
    RESDIR = Path(a.out_dir) if a.out_dir else REPO / ds["res"]
    SRCPATH = Path(a.src) if a.src else REPO / ds["src"][a.split]
    CKPT_DIR = Path(a.ckpt) if a.ckpt else REPO / ds["ckpt"]
    OUTNPZ.mkdir(parents=True, exist_ok=True)
    RESDIR.mkdir(parents=True, exist_ok=True)

    # 로스터 디렉터리(하위에 expert별 어댑터) vs 단일 어댑터(dense) 둘 다 받는다.
    # dense 디렉터리는 루트에 adapter_config.json이 있고 그 아래 checkpoint-*도 각각
    # adapter_config를 갖는다 — 하위 스캔만 하면 체크포인트를 expert로 오인한다.
    single = (CKPT_DIR / "adapter_config.json").is_file()
    if single:
        experts = [CKPT_DIR.name]
        adapter_dir = {CKPT_DIR.name: CKPT_DIR}
    else:
        experts = sorted([p.name for p in CKPT_DIR.iterdir()
                          if p.is_dir() and (p / "adapter_config.json").is_file()])
        adapter_dir = {e: CKPT_DIR / e for e in experts}
    print("experts:", experts, flush=True)
    rows = [json.loads(l) for l in open(SRCPATH, encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    ids = [str(r["id"]) for r in rows]
    print(f"src: {SRCPATH}  ({len(rows)}문제)  max_new={a.max_new_tokens} max_len={a.max_len}",
          flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()
    model = PeftModel.from_pretrained(base, str(adapter_dir[experts[0]]), adapter_name=experts[0])
    for e in experts[1:]:
        model.load_adapter(str(adapter_dir[e]), adapter_name=e)
    model.eval()

    letter_ids = []
    if MCQA:
        for ch in LETTERS:
            cand = set()
            for s in (ch, " " + ch):
                t = tok.encode(s, add_special_tokens=False)
                if t:
                    cand.add(t[0])
            letter_ids.append(sorted(cand))

    def prompt(r):
        msgs = build_baseline_prompt(
            r["instruction"],
            dataset=(r.get("dataset") or a.dataset),
            model_name=BASE,
            starter_code=r.get("starter_code"),
            domain=r.get("domain"),
        )
        if isinstance(msgs, str):        # qwen3 계열은 완성된 문자열을 반환한다
            return msgs
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    texts = [prompt(r) for r in rows]
    n, E = len(rows), len(experts)
    conf = np.zeros((n, E), np.float32)
    pred = np.zeros((n, E), np.int64)
    gen = [[None] * E for _ in range(n)]   # gen[i][ei] = 생성 텍스트

    # 전문가별 부분 저장 — 잡이 죽어도 끝난 전문가는 살아남는다.
    parts_dir = RESDIR / f"inference_{a.split}_{a.tag}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    if a.resume and MCQA:
        raise SystemExit("--resume은 conf npz를 복원할 수 없다 — MCQA에서는 쓰지 마라.")

    def load_part(e):
        """이 전문가의 부분 저장이 홀드아웃 전체를 덮을 때만 재사용한다."""
        p = parts_dir / f"{e}.jsonl"
        if not p.is_file():
            return None
        got = {}
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            got[d["id"]] = d["output"]
        return [got[i] for i in ids] if all(i in got for i in ids) else None

    for ei, e in enumerate(experts):
        if a.resume:
            cached = load_part(e)
            if cached is not None:
                for i, t in enumerate(cached):
                    gen[i][ei] = t
                print(f"  {e} resumed ({len(cached)})", flush=True)
                continue
        model.set_adapter(e)
        with open(parts_dir / f"{e}.jsonl", "w", encoding="utf-8") as pf:
            for i in range(0, n, a.batch):
                enc = tok(texts[i:i + a.batch], return_tensors="pt", padding=True,
                          truncation=True, max_length=a.max_len).to("cuda")
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                         repetition_penalty=a.repetition_penalty,
                                         output_scores=MCQA, return_dict_in_generate=True,
                                         pad_token_id=tok.pad_token_id)
                if MCQA:
                    # 첫 생성 토큰 로짓 → A-H confidence
                    lg = out.scores[0].float()                              # (B,V)
                    ll = torch.stack([lg[:, ids_].max(1).values for ids_ in letter_ids], dim=1)
                    pr = torch.softmax(ll, dim=1)
                    conf[i:i + a.batch, ei] = pr.max(1).values.cpu().numpy()
                    pred[i:i + a.batch, ei] = pr.argmax(1).cpu().numpy()
                # 생성 텍스트(프롬프트 이후만)
                new_tok = out.sequences[:, enc.input_ids.shape[1]:]
                dec = tok.batch_decode(new_tok, skip_special_tokens=True)
                for j, t in enumerate(dec):
                    gen[i + j][ei] = t.strip()
                    pf.write(json.dumps({"id": ids[i + j], "output": t.strip()},
                                        ensure_ascii=False) + "\n")
                pf.flush()
        print(f"  {e} done", flush=True)

    # 출력① binning jsonl (score_binning 호환)
    outjsonl = RESDIR / f"inference_{a.split}_{a.tag}.jsonl"
    with open(outjsonl, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            rec = {"id": ids[i], "dataset": (r.get("dataset") or a.dataset),
                   "ground_truth": r.get("ground_truth"),
                   "scoring_kind": r.get("scoring_kind", a.dataset),
                   "instruction": r.get("instruction"),
                   "expert_outputs": {e: gen[i][ei] for ei, e in enumerate(experts)}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    msg = f"done -> {outjsonl}"
    # 출력② real conf npz (MCQA 전용)
    if MCQA:
        npz = f"{a.dataset}_{a.split}_{a.tag}_conf.npz"
        np.savez(OUTNPZ / npz, conf=conf, pred=pred,
                 ids=np.array(ids), experts=np.array(experts))
        msg += f"  &  {npz}"
    print(f"{msg}  ({n}×{E})")


if __name__ == "__main__":
    main()

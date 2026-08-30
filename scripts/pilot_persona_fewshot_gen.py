#!/usr/bin/env python3
"""Persona/few-shot 재도입 파일럿 — LoRA 없이 순수 llama-3.1-8B-Instruct 프롬프팅.

배경: 배포 파이프라인(generate_lora_binning.py)은 전 expert가 build_baseline_prompt
(persona 없음)를 공유하고 LoRA 가중치만 다르다 — cap7/§4의 "다양성 없음" 결과는
"페르소나가 행동을 못 바꾼다"가 아니라 "학습 데이터 서브셋 차이 + 추론시점 무신호"를
검정한 것이었다. 이 파일럿은 그 둘을 분리한다: 같은 base 모델에 실제 진화된 persona
system_prompt를 얹었을 때(B), 그리고 각 role 자신이 푼 문제를 few-shot으로 더했을 때(C),
생성 다양성(analyze_gen_diversity.py와 동일 지표)이 조건 A(현재 배포, Jaccard 0.729)
대비 갈리는지 본다.

- 조건 B: persona system_prompt만 (few-shot 없음)
- 조건 C: persona + 자기소재 few-shot 2개 (같은 evolved 로스터의 다른 solved 문제에서 소싱,
  파일럿 문제셋과 겹치지 않음)
- 디코딩은 generate_lora_binning.py와 동일하게 greedy(do_sample=False) — A의 0.729 기준선과
  방법론을 맞춰야 비교가 성립한다.

출력: results/pilot_persona_fewshot/gen_{B,C}.jsonl (analyze_gen_diversity.py 호환 포맷:
id, expert_outputs). 채점은 하지 않는다 — 이 파일럿의 질문은 "갈리는가"이지 "맞았는가"가 아니다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
sys.path.insert(0, str(REPO / "src"))

from prompts.baseline_prompts import CODING_GEN_USER  # noqa: E402
from prompts.coding import _LLAMA_CODE_HINT  # noqa: E402
from utils.helpers import finalize_generation_output  # noqa: E402

BASE = "meta-llama/Llama-3.1-8B-Instruct"
ABL = REPO / "results/acc/seed20210111_v2/ablation"
ROSTER_PATH = REPO / "results/acc/seed20210111/roster_final.json"
OUT_DIR = REPO / "results/pilot_persona_fewshot"

N_PROBLEMS = 30
N_FEWSHOT = 2
SEED = 0


def load_roster_personas() -> dict:
    roster = json.load(open(ROSTER_PATH, encoding="utf-8"))
    return {p["id"]: p["system_prompt"] for p in roster if p["id"] != "shared"}


def load_ablation():
    gen = {json.loads(l)["id"]: json.loads(l) for l in open(ABL / "inference_test751_evolved.jsonl")}
    binned = {json.loads(l)["id"]: json.loads(l) for l in open(ABL / "inference_test751_evolved.binned.jsonl")}
    return gen, binned


def pick_pilot_problems(binned: dict, rng: random.Random, bucket: str) -> list:
    if bucket == "all-fail":
        cand = [pid for pid, b in binned.items() if b["n_solved"] == 0]
    elif bucket == "contested":
        E = len(next(iter(binned.values()))["per_expert"])
        cand = [pid for pid, b in binned.items() if 0 < b["n_solved"] < E]
    else:
        raise ValueError(f"unknown bucket: {bucket}")
    rng.shuffle(cand)
    return cand[:N_PROBLEMS]


def build_fewshot_pool(gen: dict, binned: dict, experts: list, exclude_ids: set) -> dict:
    """expert_id -> [(instruction, own_solved_code), ...] 자기소재 few-shot 후보."""
    pool = {e: [] for e in experts}
    for pid, b in binned.items():
        if pid in exclude_ids:
            continue
        for e in experts:
            if b["per_expert"].get(e) == 1:
                g = gen[pid]
                code = g["expert_outputs"].get(e)
                if code and code.strip():
                    pool[e].append((g["instruction"], code))
    return pool


def make_user_text(instruction: str, fewshot: list | None) -> str:
    base_user = CODING_GEN_USER.format(instruction=instruction)
    if not fewshot:
        return base_user
    blocks = []
    for i, (fs_instr, fs_code) in enumerate(fewshot, 1):
        blocks.append(
            f"[Example {i} — a problem you solved before, in your own style]\n"
            f"Problem:\n{fs_instr}\n\nYour solution:\n```python\n{fs_code}\n```"
        )
    preamble = (
        "Below are examples of problems you personally solved before. Solve the new "
        "problem in a similar style/approach.\n\n" + "\n\n".join(blocks)
    )
    return f"{preamble}\n\n---\n\n{base_user}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--bucket", choices=["all-fail", "contested"], default="all-fail")
    a = ap.parse_args()

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if a.bucket == "all-fail" else f"_{a.bucket}"

    personas = load_roster_personas()
    experts = sorted(personas.keys())
    gen, binned = load_ablation()

    pilot_ids = pick_pilot_problems(binned, rng, a.bucket)
    pilot_rows = [gen[pid] for pid in pilot_ids]
    fewshot_pool = build_fewshot_pool(gen, binned, experts, exclude_ids=set(pilot_ids))
    for e in experts:
        rng.shuffle(fewshot_pool[e])
    print(f"pilot problems: {len(pilot_rows)}  experts: {experts}", flush=True)
    for e in experts:
        print(f"  {e}: {len(fewshot_pool[e])} candidate few-shot solves", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()

    def prompt_text(instruction: str, persona_sys: str, fewshot: list | None) -> str:
        sys_content = persona_sys + _LLAMA_CODE_HINT
        user_content = make_user_text(instruction, fewshot)
        msgs = [{"role": "system", "content": sys_content}, {"role": "user", "content": user_content}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def run_condition(tag: str, with_fewshot: bool):
        results = {pid: {} for pid in pilot_ids}
        for e in experts:
            persona_sys = personas[e]
            fewshot = fewshot_pool[e][:N_FEWSHOT] if with_fewshot else None
            if with_fewshot and len(fewshot_pool[e]) < N_FEWSHOT:
                print(f"  WARNING {e}: only {len(fewshot_pool[e])} few-shot candidates (<{N_FEWSHOT})", flush=True)
            texts = [prompt_text(r["instruction"], persona_sys, fewshot) for r in pilot_rows]
            outs = []
            for i in range(0, len(texts), a.batch):
                chunk = texts[i:i + a.batch]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=a.max_len).to("cuda")
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                        repetition_penalty=a.repetition_penalty,
                        pad_token_id=tok.pad_token_id,
                    )
                new_tok = out[:, enc.input_ids.shape[1]:]
                dec = tok.batch_decode(new_tok, skip_special_tokens=True)
                outs.extend(t.strip() for t in dec)
            for pid, raw in zip(pilot_ids, outs):
                results[pid][e] = finalize_generation_output(raw, dataset="acc", domain="coding")
            print(f"  [{tag}] {e} done", flush=True)

        outpath = OUT_DIR / f"gen_{tag}{suffix}.jsonl"
        with open(outpath, "w", encoding="utf-8") as f:
            for pid in pilot_ids:
                r = gen[pid]
                f.write(json.dumps({
                    "id": pid,
                    "dataset": "acc",
                    "instruction": r["instruction"],
                    "expert_outputs": results[pid],
                }, ensure_ascii=False) + "\n")
        print(f"=== [{tag}] done -> {outpath}", flush=True)

    run_condition("B_persona", with_fewshot=False)
    run_condition("C_persona_fewshot", with_fewshot=True)


if __name__ == "__main__":
    main()

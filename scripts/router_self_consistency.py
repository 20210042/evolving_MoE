#!/usr/bin/env python3
"""(expert, 문제) solvability가 안정적 속성인지 시도-복권인지 직접 검정.

동기: router_sweep_max_nsolved.py에서 "결정가능"(0<n_solved<11) 172문제만 떼어봐도
hidden-state·의미임베딩 둘 다 MLP 라우터가 best-single을 못 넘었다. 이유가
(a) problem→expert 매핑은 안정적인데 아직 못 찾은 신호가 있는 것인지, 아니면
(b) 그 매핑 자체가 생성 시점(디코딩)의 우연에 가까워 원리적으로 problem-only
feature로는 예측 불가능한 것인지 구분이 안 된다.

이미 아는 결과(binned 파일, greedy/temperature=0)에 더해 temperature=0.7/top_p=0.8
(운영 기본값, [[project_eval_methodology]])로 K회 재샘플링해서, 같은 (expert,문제)의
pass율이 0/K나 K/K로 안정적으로 몰리는지 아니면 중간에 걸치는지를 본다.
중간에 많이 걸리면 (b)를 직접 뒷받침 — 라우터가 못 찾는 게 아니라 애초에 없는 것.

Usage:
  python scripts/router_self_consistency.py --dataset acc --n_problems 30 --k 5 \
      --experts c_54530 c_12606 c_6483
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import router_common as rc  # noqa: E402

REPO = rc.REPO
from evaluation.scorer import score_one  # noqa: E402
from prompts.coding import build_baseline_prompt, build_expert_prompt, build_fewshot_block  # noqa: E402
from train_sft import load_roster_personas, pick_fewshot_examples, select_expert_rows  # noqa: E402

ap = argparse.ArgumentParser()
rc.add_dataset_arg(ap, default="acc")
ap.add_argument("--ckpt", default="checkpoints/expert_sft/acc_seed20210111_v2_cap9_fewshot")
ap.add_argument("--roster_path", default="results/acc/seed20210111/roster_final.json")
ap.add_argument("--label_package", default="export/acc_binning_seed20210111_v2")
ap.add_argument("--max_n_solved", type=int, default=9)
ap.add_argument("--n_fewshot", type=int, default=2)
ap.add_argument("--experts", nargs="+", default=["c_54530", "c_12606", "c_6483"])
ap.add_argument("--n_problems", type=int, default=30)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top_p", type=float, default=0.8)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max_len", type=int, default=4096)
ap.add_argument("--max_new_tokens", type=int, default=2048)
ap.add_argument("--repetition_penalty", type=float, default=1.05)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
A = ap.parse_args()

sp = rc.spec(A.dataset)
BASE = sp.base_model
CKPT = Path(A.ckpt)
OUT = Path(A.out) if A.out else REPO / "results" / A.dataset / "router_self_consistency.md"
DEV = "cuda"
np.random.seed(A.seed)
torch.manual_seed(A.seed)

# ---- 이미 아는 greedy 결과(binned) + "결정가능" 서브셋에서 문제 표본 뽑기 ----
EX_ALL = rc.experts(sp)
test_bin = rc.load_binning(REPO / sp.eval_binned)
ids = sorted(test_bin)
S = np.array([[test_bin[i]["per_expert"].get(e, 0) for e in EX_ALL] for i in ids], np.float32)
n_solved = S.sum(1)
decidable_idx = [i for i in range(len(ids)) if 1 <= n_solved[i] < len(EX_ALL)]
rng = np.random.default_rng(A.seed)
sample_idx = rng.choice(decidable_idx, size=min(A.n_problems, len(decidable_idx)), replace=False)
sample_ids = [ids[i] for i in sample_idx]
greedy = {(e, pid): int(test_bin[pid]["per_expert"].get(e, 0)) for pid in sample_ids for e in A.experts}
print(f"결정가능 {len(decidable_idx)}문제 중 {len(sample_ids)}개 표본, expert={A.experts}, k={A.k}", flush=True)

# ---- 문제 원문 ----
src = {str(json.loads(l)["id"]): json.loads(l) for l in open(REPO / sp.src[sp.eval_split], encoding="utf-8")}

# ---- 모델/어댑터 로드 ----
print("모델 로드...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            attn_implementation="sdpa").cuda().eval()
model = PeftModel.from_pretrained(base, str(CKPT / A.experts[0]), adapter_name=A.experts[0])
for e in A.experts[1:]:
    model.load_adapter(str(CKPT / e), adapter_name=e)
model.eval()

# ---- persona+few-shot 프롬프트 (generate_lora_binning.py와 동일 재현) ----
personas = load_roster_personas(A.roster_path)
persona_approach: dict = {}
for e in A.experts:
    if e in ("shared", "common") or e not in personas:
        persona_approach[e] = (None, None)
        continue
    chosen, selected, _dsname, _is_shared = select_expert_rows(
        package_dir=A.label_package, expert_id=e, source_jsonl=None,
        seed=42, data_ratio=1.0, max_n_solved=A.max_n_solved, min_n_solved=None,
    )
    fewshot = pick_fewshot_examples(chosen, selected, A.n_fewshot)
    persona_approach[e] = (personas[e], build_fewshot_block(fewshot))
print("persona+few-shot 적용:", {e: bool(v[0]) for e, v in persona_approach.items()}, flush=True)


def prompt_text(pid: str, expert: str) -> str:
    r = src[pid]
    persona_sys, approach = persona_approach.get(expert, (None, None))
    if persona_sys:
        msgs = build_expert_prompt(
            r["instruction"], persona_sys, dataset=(r.get("dataset") or A.dataset),
            model_name=BASE, starter_code=r.get("starter_code"), approach=approach,
            domain=r.get("domain"),
        )
    else:
        msgs = build_baseline_prompt(
            r["instruction"], dataset=(r.get("dataset") or A.dataset), model_name=BASE,
            starter_code=r.get("starter_code"), domain=r.get("domain"),
        )
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def score_text(pid: str, text: str) -> int:
    r = dict(src[pid])
    r.setdefault("dataset", sp.name)
    r.setdefault("scoring_kind", sp.name)
    return 1 if score_one(r, text) > 0 else 0


# ---- (expert, 문제) × K회 temperature 샘플링 ----
results: dict[tuple[str, str], list[int]] = {}
for e in A.experts:
    model.set_adapter(e)
    jobs = [(pid, rep) for pid in sample_ids for rep in range(A.k)]
    for s in range(0, len(jobs), A.batch):
        chunk = jobs[s:s + A.batch]
        texts = [prompt_text(pid, e) for pid, _ in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=A.max_len).to(DEV)
        with torch.no_grad():
            seq = model.generate(**enc, max_new_tokens=A.max_new_tokens, do_sample=True,
                                 temperature=A.temperature, top_p=A.top_p,
                                 repetition_penalty=A.repetition_penalty,
                                 pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(seq[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for (pid, rep), text in zip(chunk, dec):
            pass_ = score_text(pid, text.strip())
            results.setdefault((e, pid), []).append(pass_)
    n_done = sum(len(v) for k, v in results.items() if k[0] == e)
    print(f"  {e}: {n_done}건 생성 완료", flush=True)

# ---- 집계: pass율이 0/K, K/K(안정) vs 중간(불안정) ----
stable_0, stable_k, mixed = 0, 0, 0
mismatch_vs_greedy = 0
lines_detail = []
for e in A.experts:
    for pid in sample_ids:
        outs = results[(e, pid)]
        k_pass = sum(outs)
        g = greedy[(e, pid)]
        tag = "stable-0" if k_pass == 0 else ("stable-K" if k_pass == A.k else "MIXED")
        if tag == "stable-0":
            stable_0 += 1
        elif tag == "stable-K":
            stable_k += 1
        else:
            mixed += 1
        greedy_consistent = (g == 1) == (k_pass > 0)
        if not greedy_consistent:
            mismatch_vs_greedy += 1
        lines_detail.append((e, pid, g, k_pass, A.k, tag, greedy_consistent))

total = len(A.experts) * len(sample_ids)
print(f"\n총 {total} (expert,문제) 쌍 — stable-0: {stable_0}  stable-K: {stable_k}  "
      f"MIXED(중간): {mixed}  ({100*mixed/total:.1f}%)", flush=True)
print(f"greedy 라벨과 방향 불일치(원래 0인데 K회중 1번이상 성공, 또는 그 반대): "
      f"{mismatch_vs_greedy}/{total} ({100*mismatch_vs_greedy/total:.1f}%)", flush=True)

lines = [
    "# ACC 라우터 self-consistency 체크 — solvability가 안정적 속성인가, 시도-복권인가",
    "",
    f"- expert: {A.experts} · 문제 표본: {len(sample_ids)}개(결정가능 {len(decidable_idx)}개 중) · "
    f"K={A.k} (temperature={A.temperature}, top_p={A.top_p})",
    f"- **총 {total}쌍 중 MIXED(0<pass<{A.k}, 시도마다 결과가 갈림): {mixed}개 ({100*mixed/total:.1f}%)**",
    f"- greedy 라벨과 방향 불일치: {mismatch_vs_greedy}개 ({100*mismatch_vs_greedy/total:.1f}%)",
    "",
    "| expert | 문제 | greedy라벨 | k회 중 pass | 판정 | greedy와 방향일치 |",
    "|---|---|---:|---:|---|---|",
]
for e, pid, g, k_pass, k, tag, ok in lines_detail:
    lines.append(f"| {e} | {pid[:40]} | {g} | {k_pass}/{k} | {tag} | {'O' if ok else 'X'} |")
txt = "\n".join(lines) + "\n"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(txt, encoding="utf-8")
print(f"\nsaved -> {OUT}")

#!/usr/bin/env python3
"""End-to-end top-1 배포 평가 — 병합 없이, 라우터가 고른 어댑터 1개로 실제 생성·채점.

moe_deploy_sweep.py(top-2 0.5 병합)는 전 expert가 같은 프롬프트를 공유한다는 전제라
persona+few-shot expert마다 프롬프트가 다른 조건(acc Evolved)엔 못 쓴다 — 두 expert를
병합했을 때 "어느 persona로 프롬프트를 만들지"가 정의되지 않는다. 그래서 top-1(병합 없이
라우터가 고른 expert 1명이 자기 프롬프트로 그대로 생성)로 간다. 대신 top-1은 top-2보다
oracle에 못 미칠 걸 예상하고 들어간다(QASC 선례, [[project-router-top1-limit]]).

Usage:
  python scripts/moe_deploy_top1.py --dataset acc \
      --ckpt checkpoints/expert_sft/acc_seedhp_v2_cap9 \
      --binned results/acc/seed20210111_v2/ablation/inference_test751_hp.binned.jsonl \
      --dense_acc 15.05 --label "Human-prior MoE" \
      --out results/acc/seed20210111_v2/ablation/deploy_top1_hp.md

  # persona+few-shot 조건(Evolved)은 roster_path/label_package를 추가:
  python scripts/moe_deploy_top1.py --dataset acc \
      --ckpt checkpoints/expert_sft/acc_seed20210111_v2_cap9_fewshot \
      --binned results/acc/seed20210111_v2/ablation/inference_test751_evolved_fewshot.binned.jsonl \
      --roster_path results/acc/seed20210111/roster_final.json \
      --label_package export/acc_binning_seed20210111_v2 --max_n_solved 9 --n_fewshot 2 \
      --dense_acc 15.05 --label "Evolved MoE (persona+fewshot)" \
      --out results/acc/seed20210111_v2/ablation/deploy_top1_evolved_fewshot.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import router_common as rc  # noqa: E402

REPO = rc.REPO
from evaluation.scorer import score_one  # noqa: E402
from prompts.coding import build_baseline_prompt, build_expert_prompt, build_fewshot_block  # noqa: E402
from train_sft import load_roster_personas, pick_fewshot_examples, select_expert_rows  # noqa: E402

ap = argparse.ArgumentParser()
rc.add_dataset_arg(ap, default="acc")
ap.add_argument("--ckpt", required=True, help="로스터 어댑터 디렉터리(하위에 expert별 폴더)")
ap.add_argument("--binned", required=True, help="라우터 학습 라벨 = eval split의 per-expert solve jsonl")
ap.add_argument("--dense_acc", type=float, required=True, help="Dense SFT 앵커 pass@1(%%), 이미 아는 값")
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="MoE (top-1)")
ap.add_argument("--roster_path", default=None, help="persona+few-shot 조건에만 지정")
ap.add_argument("--label_package", default=None, help="--roster_path와 함께 필요 (few-shot 소싱)")
ap.add_argument("--n_fewshot", type=int, default=2)
ap.add_argument("--max_n_solved", type=int, default=None)
ap.add_argument("--min_n_solved", type=int, default=None)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max_len", type=int, default=4096)
ap.add_argument("--max_new_tokens", type=int, default=2048)
ap.add_argument("--repetition_penalty", type=float, default=1.05)
A = ap.parse_args()

sp = rc.spec(A.dataset)
BASE = sp.base_model
CKPT = Path(A.ckpt)
BINNED = Path(A.binned)
OUT = Path(A.out)
DEV = "cuda"
np.random.seed(0)
torch.manual_seed(0)

# ---- 정답지: 이 조건의 per-expert solve 매트릭스 (라우터 학습 타깃) ----
binned = [json.loads(l) for l in open(BINNED, encoding="utf-8")]
bids = [str(r["id"]) for r in binned]
EX = sorted(binned[0]["per_expert"])
S = np.array([[r["per_expert"].get(e, 0) for e in EX] for r in binned], np.float32)
N, E = S.shape
best_single, oracle_union = rc.baselines(S)
print(f"experts({E}): {EX}  best-single={best_single:.2f}  oracle-union={oracle_union:.2f}", flush=True)

# ---- 문제 원문 (프롬프트/채점용) ----
src = {str(json.loads(l)["id"]): json.loads(l) for l in open(REPO / sp.src[sp.eval_split], encoding="utf-8")}

# ---- 라우터 입력 특징: base LLM hidden-state (extract_hidden_states.py로 미리 추출) ----
HS = rc.align(rc.feat_path(sp, sp.eval_split, "hs_mean"), rc.feat_path(sp, sp.eval_split, "hs_ids"),
              rc.load_binning(BINNED), EX, order=bids)[1]


def mlp(d, hid=512, drop=0.3):
    return nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, E))


def cv_top1(X, ep=120, seeds=(0, 1, 2), folds=5):
    """5-fold CV top-1 argmax — 각 문제는 자기가 안 낀 fold로 학습된 라우터가 고른다."""
    Xa = X.astype(np.float32)
    St = torch.tensor(S)
    idx = np.arange(N)
    np.random.default_rng(0).shuffle(idx)
    fold = np.array_split(idx, folds)
    logit = np.zeros((N, E), np.float32)
    for f in range(folds):
        te = fold[f]
        tr = np.concatenate([fold[g] for g in range(folds) if g != f])
        mu, sd = Xa[tr].mean(0, keepdims=True), Xa[tr].std(0, keepdims=True) + 1e-6
        Xtr = torch.tensor((Xa[tr] - mu) / sd).to(DEV)
        Xte = torch.tensor((Xa[te] - mu) / sd).to(DEV)
        Str = St[tr].to(DEV)
        segs = []
        for s in seeds:
            torch.manual_seed(s)
            net = mlp(Xa.shape[1]).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
            lf = nn.BCEWithLogitsLoss()
            for _ in range(ep):
                p = torch.randperm(len(Xtr), device=DEV)
                for i in range(0, len(Xtr), 256):
                    b = p[i:i + 256]
                    opt.zero_grad()
                    lf(net(Xtr[b]), Str[b]).backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                segs.append(net(Xte).cpu().numpy())
        logit[te] = np.mean(segs, 0)
    return logit.argmax(1)


print("top-1 라우터 CV 학습...", flush=True)
rng = np.random.default_rng(0)
methods = {
    "random-1": rng.integers(0, E, N),
    "best-single (fixed)": np.full(N, int(S.mean(0).argmax())),
    "MLP hidden-state (top-1)": cv_top1(HS),
    "oracle top-1": np.array([int(np.flatnonzero(S[i])[0]) if S[i].sum() > 0
                               else int(S.mean(0).argmax()) for i in range(N)]),
}

# ==== 모델/어댑터 로드 (1회) ====
print("모델 로드...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            attn_implementation="sdpa").cuda().eval()
model = PeftModel.from_pretrained(base, str(CKPT / EX[0]), adapter_name=EX[0])
for e in EX[1:]:
    model.load_adapter(str(CKPT / e), adapter_name=e)
model.eval()

# ---- persona+few-shot 프롬프트 (roster_path 지정 시, generate_lora_binning.py와 동일 재현) ----
persona_approach: dict = {}
if A.roster_path:
    if not A.label_package:
        raise SystemExit("--roster_path와 함께 --label_package가 필요합니다 (few-shot 소싱).")
    personas = load_roster_personas(A.roster_path)
    for e in EX:
        if e in ("shared", "common") or e not in personas:
            persona_approach[e] = (None, None)
            continue
        chosen, selected, _dsname, _is_shared = select_expert_rows(
            package_dir=A.label_package, expert_id=e, source_jsonl=None,
            seed=42, data_ratio=1.0, max_n_solved=A.max_n_solved, min_n_solved=A.min_n_solved,
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


# ---- 방법별로 필요한 (expert, 문제) 생성 요청 모으기 (expert 단위로 배치) ----
need = {e: [] for e in EX}
method_pick = {}
for m, picks in methods.items():
    method_pick[m] = [EX[int(picks[i])] for i in range(N)]
    for i in range(N):
        need[method_pick[m][i]].append(i)
for e in EX:
    need[e] = sorted(set(need[e]))

memo: dict[tuple[str, int], str] = {}
total_gen = sum(len(v) for v in need.values())
print(f"생성 시작: 어댑터 {sum(1 for v in need.values() if v)}개, 총 (expert,문제) {total_gen}건", flush=True)
for e, idlist in need.items():
    if not idlist:
        continue
    model.set_adapter(e)
    for s in range(0, len(idlist), A.batch):
        chunk = idlist[s:s + A.batch]
        texts = [prompt_text(bids[i], e) for i in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=A.max_len).to(DEV)
        with torch.no_grad():
            seq = model.generate(**enc, max_new_tokens=A.max_new_tokens, do_sample=False,
                                 repetition_penalty=A.repetition_penalty,
                                 pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(seq[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for i, t in zip(chunk, dec):
            memo[(e, i)] = t.strip()
    print(f"  {e}: {len(idlist)} gen", flush=True)


def score_text(i: int, text: str) -> int:
    r = dict(src[bids[i]])
    r.setdefault("dataset", sp.name)
    r.setdefault("scoring_kind", sp.name)
    return 1 if score_one(r, text) > 0 else 0


# ---- 방법별 실제 배포 정확도 ----
results = {}
for m in methods:
    accs = [score_text(i, memo[(method_pick[m][i], i)]) for i in range(N)]
    results[m] = 100 * np.mean(accs)
    print(f"  [{m}] {results[m]:.2f}%", flush=True)

order = ["random-1", "best-single (fixed)", "MLP hidden-state (top-1)", "oracle top-1"]
lines = [
    f"# {sp.name.upper()} End-to-End 배포: {A.label} (top-1, 병합 없음) vs Dense SFT",
    "",
    f"- **앵커 (Dense SFT baseline)**: **{A.dense_acc:.2f}%**",
    f"- 조건: **{A.label}** (experts={E}) · 참조: best-single(solve) {best_single:.1f}% · oracle-union {oracle_union:.1f}%",
    f"- 방식: 라우터가 고른 어댑터 1개만 활성화(병합 없음) → {N}문제 실제 생성 → EM 채점",
    "",
    "| 라우팅 방법 | 배포 정확도(%) | vs Dense | vs best-single(solve) |",
    "|---|---|---|---|",
]
for m in order:
    lines.append(f"| {m} | {results[m]:.2f} | {results[m]-A.dense_acc:+.2f} | {results[m]-best_single:+.2f} |")
txt = "\n".join(lines) + "\n"
OUT.write_text(txt, encoding="utf-8")
print("\n" + txt)
print(f"saved -> {OUT}")

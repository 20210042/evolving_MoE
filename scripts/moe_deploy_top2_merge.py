#!/usr/bin/env python3
"""End-to-end top-2 배포 (코딩) — 라우터가 고른 상위 2 어댑터를 0.5/0.5 병합해 한 번 생성·채점.

과업 3번의 코딩 도메인 실행. QASC는 moe_deploy_sweep.py로 이미 수행됐지만(전 expert가 같은
프롬프트를 공유한다는 전제), 코딩 Evolved 조건은 expert마다 persona+few-shot 프롬프트가 달라
"두 어댑터를 병합하면 어느 persona로 프롬프트를 만드나"가 정의되지 않는다.

확정 규약: **어댑터는 top-2를 0.5/0.5 선형 병합, 프롬프트는 top-1 expert의 persona+few-shot**.
비대칭이지만 정의가 명확하고 기존 top-1 배포(moe_deploy_top1.py)와 직접 비교된다.

병합 어댑터는 (정렬된 pair) 단위로 방법 무관 동일 → pair 단위로 만들고, 프롬프트 주인별로
묶어 생성한다. 배선 앵커: weights=[1,0] 병합이 그 expert 단독 생성과 일치해야 한다(--wiring_check).

Usage:
  python scripts/moe_deploy_top2_merge.py --dataset acc \
      --ckpt checkpoints/expert_sft/acc_seed20210111_v2_cap9_fewshot \
      --binned results/acc/seed20210111_v2/ablation/inference_test751_evolved_fewshot.binned.jsonl \
      --roster_path results/acc/seed20210111/roster_final.json \
      --label_package export/acc_binning_seed20210111_v2 --max_n_solved 9 --n_fewshot 2 \
      --dense_acc 15.05 --top1_acc 14.25 --label "Evolved MoE (persona+fewshot)" \
      --out results/acc/seed20210111_v2/ablation/deploy_top2_evolved_fewshot.md
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
ap.add_argument("--ckpt", required=True)
ap.add_argument("--binned", required=True)
ap.add_argument("--dense_acc", type=float, required=True, help="Dense SFT 앵커 pass@1(%%)")
ap.add_argument("--top1_acc", type=float, default=None, help="같은 조건 top-1 MLP 라우터 수치(비교용)")
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="MoE (top-2 merge)")
ap.add_argument("--roster_path", default=None, help="persona+few-shot 조건에만 지정")
ap.add_argument("--label_package", default=None)
ap.add_argument("--n_fewshot", type=int, default=2)
ap.add_argument("--max_n_solved", type=int, default=None)
ap.add_argument("--min_n_solved", type=int, default=None)
ap.add_argument("--weights", default="0.5,0.5")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max_len", type=int, default=4096)
ap.add_argument("--max_new_tokens", type=int, default=2048)
ap.add_argument("--repetition_penalty", type=float, default=1.05)
ap.add_argument("--wiring_check", type=int, default=4,
                help="weights=[1,0] 병합이 단독 어댑터와 같은 출력을 내는지 확인할 문제 수 (0=생략)")
A = ap.parse_args()

sp = rc.spec(A.dataset)
BASE = sp.base_model
CKPT = Path(A.ckpt)
OUT = Path(A.out)
DEV = "cuda"
W1, W2 = [float(x) for x in A.weights.split(",")]
np.random.seed(0)
torch.manual_seed(0)

# ---- 정답지: 이 조건의 per-expert solve 매트릭스 ----
binned = [json.loads(l) for l in open(A.binned, encoding="utf-8")]
bids = [str(r["id"]) for r in binned]
EX = sorted(binned[0]["per_expert"])
S = np.array([[r["per_expert"].get(e, 0) for e in EX] for r in binned], np.float32)
N, E = S.shape
best_single, oracle_union = rc.baselines(S)
print(f"experts({E}): {EX}  best-single={best_single:.2f}  oracle-union={oracle_union:.2f}", flush=True)

src = {str(json.loads(l)["id"]): json.loads(l) for l in open(REPO / sp.src[sp.eval_split], encoding="utf-8")}
HS = rc.align(rc.feat_path(sp, sp.eval_split, "hs_mean"), rc.feat_path(sp, sp.eval_split, "hs_ids"),
              rc.load_binning(A.binned), EX, order=bids)[1]


def mlp(d, hid=512, drop=0.3):
    return nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, E))


def cv_logits(X, ep=120, seeds=(0, 1, 2), folds=5):
    """moe_deploy_top1.py의 cv_top1과 동일한 라우터. argmax 대신 로짓을 그대로 준다."""
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
    return logit


print("top-2 라우터 CV 학습...", flush=True)
rng = np.random.default_rng(0)
L = cv_logits(HS)
mlp_top2 = np.argsort(-L, axis=1)[:, :2]

rand2 = np.array([rng.choice(E, size=2, replace=False) for _ in range(N)])

# oracle top-2: 실제로 푸는 expert를 최대 2명까지 채우고, 모자라면 평균 최고 expert로 채운다.
fallback = int(S.mean(0).argmax())
oracle2 = []
for i in range(N):
    sol = list(np.flatnonzero(S[i]))
    if len(sol) >= 2:
        oracle2.append([int(sol[0]), int(sol[1])])
    elif len(sol) == 1:
        alt = fallback if fallback != sol[0] else (fallback + 1) % E
        oracle2.append([int(sol[0]), int(alt)])
    else:
        oracle2.append([fallback, (fallback + 1) % E])
oracle2 = np.array(oracle2)

methods = {
    "random-2": rand2,
    "MLP hidden-state (top-2)": mlp_top2,
    "oracle top-2": oracle2,
}

# ==== 모델/어댑터 로드 ====
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

# ---- persona+few-shot 프롬프트 (moe_deploy_top1.py와 동일 재현) ----
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
        persona_approach[e] = (personas[e], build_fewshot_block(
            pick_fewshot_examples(chosen, selected, A.n_fewshot)))
    print("persona+few-shot 적용:", {e: bool(v[0]) for e, v in persona_approach.items()}, flush=True)


def prompt_text(pid: str, expert: str) -> str:
    """프롬프트 주인 = top-1 expert."""
    r = src[pid]
    persona_sys, approach = persona_approach.get(expert, (None, None))
    if persona_sys:
        msgs = build_expert_prompt(
            r["instruction"], persona_sys, dataset=(r.get("dataset") or A.dataset),
            model_name=BASE, starter_code=r.get("starter_code"), approach=approach,
            domain=r.get("domain"))
    else:
        msgs = build_baseline_prompt(
            r["instruction"], dataset=(r.get("dataset") or A.dataset), model_name=BASE,
            starter_code=r.get("starter_code"), domain=r.get("domain"))
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def gen(idlist, owner):
    out = {}
    for s in range(0, len(idlist), A.batch):
        chunk = idlist[s:s + A.batch]
        enc = tok([prompt_text(bids[i], owner) for i in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=A.max_len).to(DEV)
        with torch.no_grad():
            seq = model.generate(**enc, max_new_tokens=A.max_new_tokens, do_sample=False,
                                 repetition_penalty=A.repetition_penalty,
                                 pad_token_id=tok.pad_token_id)
        for i, t in zip(chunk, tok.batch_decode(seq[:, enc.input_ids.shape[1]:],
                                                skip_special_tokens=True)):
            out[i] = t.strip()
    return out


def merged(pair, w):
    """정렬된 pair를 w로 선형 병합한 임시 어댑터를 활성화. 이름 반환(사용 후 삭제할 것)."""
    name = f"_m_{pair[0]}__{pair[1]}"
    if name in getattr(model, "peft_config", {}):
        model.delete_adapter(name)
    model.add_weighted_adapter(adapters=list(pair), weights=list(w),
                               adapter_name=name, combination_type="linear")
    model.set_adapter(name)
    return name


# ---- 배선 앵커: weights=[1,0] 병합 == 단독 어댑터 ----
wiring = None
if A.wiring_check > 0 and E >= 2:
    ids = list(range(min(A.wiring_check, N)))
    e1, e2 = EX[0], EX[1]
    model.set_adapter(e1)
    solo = gen(ids, e1)
    nm = merged((e1, e2), (1.0, 0.0))
    mm = gen(ids, e1)
    model.delete_adapter(nm)
    same = sum(1 for i in ids if solo[i] == mm[i])
    wiring = (same, len(ids))
    print(f"[배선 앵커] weights=[1,0] 병합 == 단독 어댑터: {same}/{len(ids)} 일치", flush=True)
    if same != len(ids):
        print("  ⚠️ 불일치 — 병합 경로를 신뢰할 수 없다. 결과 해석 주의.", flush=True)

# ---- (pair, 프롬프트주인) 단위로 생성 요청 모으기 ----
need: dict = {}
pick = {}
for m, p2 in methods.items():
    pick[m] = [(EX[int(p2[i][0])], EX[int(p2[i][1])]) for i in range(N)]
    for i in range(N):
        top1, top2 = pick[m][i]
        key = (tuple(sorted((top1, top2))), top1)
        need.setdefault(key, set()).add(i)

memo: dict = {}
print(f"생성 시작: 조합 {len(need)}개, 총 {sum(len(v) for v in need.values())}건", flush=True)
by_pair: dict = {}
for (pair, owner), ids in need.items():
    by_pair.setdefault(pair, []).append((owner, sorted(ids)))
for pair, groups in by_pair.items():
    nm = merged(pair, (W1, W2))
    for owner, ids in groups:
        memo.update({(pair, owner, i): t for i, t in gen(ids, owner).items()})
    model.delete_adapter(nm)
    print(f"  {pair}: {sum(len(g[1]) for g in groups)} gen", flush=True)


def score_text(i: int, text: str) -> int:
    r = dict(src[bids[i]])
    r.setdefault("dataset", sp.name)
    r.setdefault("scoring_kind", sp.name)
    return 1 if score_one(r, text) > 0 else 0


results = {}
for m in methods:
    accs = []
    for i in range(N):
        top1, top2 = pick[m][i]
        accs.append(score_text(i, memo[(tuple(sorted((top1, top2))), top1, i)]))
    results[m] = 100 * float(np.mean(accs))
    print(f"  [{m}] {results[m]:.2f}%", flush=True)

lines = [
    f"# {sp.name.upper()} End-to-End 배포: {A.label} — top-2 어댑터 {W1}/{W2} 병합",
    "",
    f"- **앵커 (Dense SFT baseline)**: **{A.dense_acc:.2f}%**",
    f"- 조건: **{A.label}** (experts={E}) · 참조: best-single(solve) {best_single:.1f}% · "
    f"oracle-union {oracle_union:.1f}%",
    f"- 규약: 어댑터는 top-2를 {W1}/{W2} 선형 병합, **프롬프트는 top-1 expert의 persona+few-shot** "
    f"→ {N}문제 실제 생성 → EM 채점",
]
if wiring:
    lines.append(f"- 배선 앵커: `weights=[1,0]` 병합 == 단독 어댑터 출력 **{wiring[0]}/{wiring[1]} 일치**")
lines += ["", "| 라우팅 방법 | 배포 정확도(%) | vs Dense | vs best-single(solve) |", "|---|---|---|---|"]
for m in ["random-2", "MLP hidden-state (top-2)", "oracle top-2"]:
    lines.append(f"| {m} | {results[m]:.2f} | {results[m]-A.dense_acc:+.2f} | "
                 f"{results[m]-best_single:+.2f} |")
if A.top1_acc is not None:
    lines += ["", f"- 같은 조건 top-1(병합 없음) MLP 라우터: **{A.top1_acc:.2f}%** → "
              f"top-2 병합 Δ = **{results['MLP hidden-state (top-2)']-A.top1_acc:+.2f}pp**"]
txt = "\n".join(lines) + "\n"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(txt, encoding="utf-8")
print("\n" + txt)
print(f"saved -> {OUT}")

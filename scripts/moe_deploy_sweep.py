#!/usr/bin/env python3
"""End-to-end 배포 스윕 — 라우팅 방법별 top-2 어댑터 0.5/0.5 병합 → 실제 생성 → 채점.

각 방법이 문제별 top-2 expert를 고르면, 그 둘을 linear 0.5/0.5로 한 어댑터로 병합해
'한 번' 생성한 답을 채점(EM). 앵커 = Jongbin 공유 dense fine-tuned Llama3 baseline(87.15%).
pair별 병합 어댑터는 방법 무관 동일 → (pair,문제) 단위 메모이즈로 중복 생성 제거.

출력: results/qasc/seed20210211/deploy_sweep_vs_dense.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
sys.path.insert(0, str(REPO / "src"))
from prompts import baseline_prompts as bp  # noqa: E402
from evaluation.scorer import score_qasc_item  # noqa: E402

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default=str(REPO / "checkpoints/expert_sft/qasc_seed20210211_cap10"))
_ap.add_argument("--binned", default=str(REPO / "results/qasc/seed20210211/inference_validation_lora13.binned.jsonl"))
_ap.add_argument("--conf", default=str(REPO / "results/embed_viz_test/qasc_validation_lora_conf.npz"))
_ap.add_argument("--out", default=str(REPO / "results/qasc/seed20210211/deploy_sweep_vs_dense.md"))
_ap.add_argument("--label", default="Evolved MoE", help="조건 이름(표 제목)")
_A = _ap.parse_args()

O = REPO / "results/embed_viz_test"
BINNED = Path(_A.binned)
SRC = REPO / "export/qasc/qasc_validation.jsonl"
DENSE = Path("/home/jaehoonjeong/data/MetaAgentEvolution_Release/"
             "qasc_sft_llama3_finetuned_qasc_baseline_eval300_ep9_baseline_208314.jsonl")
BASE = "meta-llama/Llama-3.1-8B-Instruct"
CKPT = Path(_A.ckpt)
OUT = Path(_A.out)
BATCH, MAXLEN, MAXNEW = 48, 1024, 8
DEV = "cuda"
np.random.seed(0)
torch.manual_seed(0)


def ids_json(p):
    return [str(x) for x in json.load(open(p))]


# ---- 정답지 / 소스 ----
binned = [json.loads(l) for l in open(BINNED)]
bids = [str(r["id"]) for r in binned]
EX = sorted(binned[0]["per_expert"])
S = np.array([[r["per_expert"].get(e, 0) for e in EX] for r in binned], np.float32)
N, E = S.shape
src = {str(json.loads(l)["id"]): json.loads(l) for l in open(SRC, encoding="utf-8")}
best_single = 100 * S.mean(0).max()
oracle_union = 100 * (S.sum(1) > 0).mean()
dense_rows = [json.loads(l) for l in open(DENSE, encoding="utf-8")]
dense_acc = 100 * np.mean([float(r["pass_score"]) > 0 for r in dense_rows])


def align(fp, ip):
    F = np.load(fp)
    fid = ids_json(ip)
    idx = {p: i for i, p in enumerate(fid)}
    return np.array([F[idx[p]] for p in bids], np.float32)


# ---- feature ----
cd = np.load(_A.conf, allow_pickle=True)
c_ids = [str(x) for x in cd["ids"]]
c_ex = [str(x) for x in cd["experts"]]
ci = {p: i for i, p in enumerate(c_ids)}
cj = {e: j for j, e in enumerate(c_ex)}
CONF = np.array([[cd["conf"][ci[p], cj[e]] for e in EX] for p in bids], np.float32)
PRED = np.array([[cd["pred"][ci[p], cj[e]] for e in EX] for p in bids], np.int64)
HS = align(O / "qasc_validation_hs_mean.npy", O / "qasc_validation_hs_ids.json")
ANS = align(O / "qasc_validation_ansprob.npy", O / "qasc_validation_ansprob_ids.json")
EMB = align(O / "qasc_val_emb.npy", REPO / "export/qasc/qasc_validation.jsonl") \
    if False else None
# emb는 qasc_validation.jsonl 순서
emb_raw = np.load(O / "qasc_val_emb.npy")
emb_ids = [str(json.loads(l)["id"]) for l in open(SRC, encoding="utf-8")]
eidx = {p: i for i, p in enumerate(emb_ids)}
EMB = np.array([emb_raw[eidx[p]] for p in bids], np.float32)


def top2(score):
    return score.argsort(1)[:, -2:][:, ::-1]


def zc(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)


def rc(X):
    return np.argsort(np.argsort(X, 0), 0).astype(np.float32)


# ---- 학습 라우터 5-fold CV logit ----
def mlp(d, hid=512, drop=0.3):
    return nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, E))


def cv_logits(X, ep=120, seeds=(0, 1, 2), folds=5):
    Xa = X.astype(np.float32)
    St = torch.tensor(S)
    idx = np.arange(N)
    np.random.default_rng(0).shuffle(idx)
    fold = np.array_split(idx, folds)
    out = np.zeros((N, E), np.float32)
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
        out[te] = np.mean(segs, 0)
    return out


print("학습 라우터 CV 중...", flush=True)
mlp_log = {"MLP hidden-state": cv_logits(HS), "MLP encoder-emb": cv_logits(EMB),
           "MLP answer-prob": cv_logits(ANS), "MLP confidence": cv_logits(CONF),
           "MLP hs+conf": cv_logits(np.concatenate([HS, CONF], 1))}

# pred-agreement
agree = np.zeros((N, E), np.float32)
for i in range(N):
    v = np.zeros(8, np.float32)
    for e in range(E):
        v[PRED[i, e]] += CONF[i, e]
    plur = v.argmax()
    for e in range(E):
        agree[i, e] = (1.0 if PRED[i, e] == plur else 0.0) + 0.01 * CONF[i, e]

prior = S.mean(0, keepdims=True)
rng = np.random.default_rng(0)
rand2 = np.array([rng.choice(E, 2, replace=False) for _ in range(N)])
oracle_score = S * 10 + CONF

# ---- 방법 → top-2 (N,2) 픽 ----
methods = {
    "random-2": rand2,
    "confidence (raw)": top2(CONF),
    "confidence (z-norm)": top2(zc(CONF)),
    "confidence (rank)": top2(rc(CONF)),
    "confidence + prior": top2(zc(CONF) + 3.0 * zc(np.repeat(prior, N, 0))),
    "pred-agreement": top2(agree),
    "MLP hidden-state": top2(mlp_log["MLP hidden-state"]),
    "MLP encoder-emb": top2(mlp_log["MLP encoder-emb"]),
    "MLP answer-prob": top2(mlp_log["MLP answer-prob"]),
    "MLP confidence": top2(mlp_log["MLP confidence"]),
    "MLP hs+conf": top2(mlp_log["MLP hs+conf"]),
    "oracle top-2": top2(oracle_score),
}

# ==== 모델 로드 (1회) ====
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


def prompt(i):
    r = src[i]
    msgs = [{"role": "system", "content": bp.QASC_GEN_SYSTEM},
            {"role": "user", "content": bp.QASC_GEN_USER.format(instruction=r["instruction"])}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


pair_adapter = {}   # sorted pair(tuple) -> adapter name


def adapter_for(pair):
    key = tuple(sorted(pair))
    if key in pair_adapter:
        return pair_adapter[key]
    if key[0] == key[1]:
        name = EX[key[0]]                       # 동일 expert면 병합 불필요
    else:
        names = [EX[key[0]], EX[key[1]]]
        name = "m__" + "__".join(names)
        model.add_weighted_adapter(names, [0.5, 0.5], adapter_name=name, combination_type="linear")
    pair_adapter[key] = name
    return name


# ---- 필요한 (adapter, 문제) 수집 ----
need = defaultdict(set)     # adapter_name -> set(problem idx)
method_pair = {}            # method -> list[adapter_name] per problem
for m, picks in methods.items():
    row = []
    for i in range(N):
        an = adapter_for((int(picks[i][0]), int(picks[i][1])))
        row.append(an)
        need[an].add(i)
    method_pair[m] = row

# ---- 생성 (adapter별 배치) ----
memo = {}   # (adapter_name, i) -> answer text
print(f"생성 시작: 어댑터 {len(need)}개, 총 (adapter,문제) {sum(len(v) for v in need.values())}건", flush=True)
for an, idxset in need.items():
    model.set_adapter(an)
    idlist = sorted(idxset)
    for s in range(0, len(idlist), BATCH):
        chunk = idlist[s:s + BATCH]
        enc = tok([prompt(bids[i]) for i in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to("cuda")
        with torch.no_grad():
            seq = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(seq[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for i, t in zip(chunk, dec):
            memo[(an, i)] = t.strip()
    print(f"  {an}: {len(idlist)} gen", flush=True)


def score_text(i, text):
    r = src[bids[i]]
    item = {"id": bids[i], "dataset": "qasc", "ground_truth": r["ground_truth"],
            "scoring_kind": "qasc", "instruction": r.get("instruction", "")}
    return 1 if score_qasc_item(item, text) > 0 else 0


# ---- 방법별 정확도 ----
results = {}
for m in methods:
    accs = [score_text(i, memo[(method_pair[m][i], i)]) for i in range(N)]
    results[m] = 100 * np.mean(accs)
    print(f"  [{m}] {results[m]:.1f}%", flush=True)

# ==== 출력 ====
order = ["random-2", "confidence (raw)", "confidence (z-norm)", "confidence (rank)",
         "confidence + prior", "pred-agreement", "MLP hidden-state", "MLP encoder-emb",
         "MLP answer-prob", "MLP confidence", "MLP hs+conf", "oracle top-2"]
lines = [
    f"# QASC End-to-End 배포: {_A.label} (top-2 0.5 병합) vs Dense fine-tuned Llama3",
    "",
    f"- **앵커 (Dense fine-tuned Llama3, ep9 baseline)**: **{dense_acc:.2f}%** ({DENSE.name})",
    f"- 조건: **{_A.label}** (experts={E}) · 참조: best-single(solve) {best_single:.1f}% · oracle-union {oracle_union:.1f}%",
    f"- 방식: 라우팅 방법이 고른 top-2 어댑터를 linear 0.5/0.5로 병합 → 926문제 실제 생성 → EM 채점",
    "",
    "| 라우팅 방법 | 배포 정확도(%) | vs Dense(87.15) |",
    "|---|---|---|",
]
for m in order:
    lines.append(f"| {m} | {results[m]:.1f} | {results[m]-dense_acc:+.1f} |")
txt = "\n".join(lines) + "\n"
OUT.write_text(txt, encoding="utf-8")
print("\n" + txt)
print(f"saved -> {OUT}")

#!/usr/bin/env python3
"""End-to-end 배포 스윕 — 라우팅 방법별 top-2 어댑터 0.5/0.5 병합 → 실제 생성 → 채점.

각 방법이 문제별 top-2 expert를 고르면, 그 둘을 linear 0.5/0.5로 한 어댑터로 병합해
'한 번' 생성한 답을 채점(EM). 앵커 = Jongbin 공유 dense fine-tuned Llama3 baseline(87.15%).
pair별 병합 어댑터는 방법 무관 동일 → (pair,문제) 단위 메모이즈로 중복 생성 제거.

라우팅 방법 12개가 전부 이 파일에 있다(methods dict). confidence/answer-prob 계열은
MCQA 전용이라 오픈 QA 데이터셋에서는 자동으로 목록에서 빠진다.

Usage: python scripts/moe_deploy_sweep.py --dataset qasc [--ckpt ... --out ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

REPO = rc.REPO
from evaluation.scorer import score_one  # noqa: E402

import argparse

_ap = argparse.ArgumentParser()
rc.add_dataset_arg(_ap)
_ap.add_argument("--ckpt", default=None)
_ap.add_argument("--binned", required=True,
                 help="배포에 쓸 solve 매트릭스 (per-expert binned jsonl)")
_ap.add_argument("--conf", default=None, help="expert confidence npz (MCQA 전용)")
_ap.add_argument("--dense", required=True, help="앵커로 쓸 dense SFT baseline jsonl")
_ap.add_argument("--out", required=True, help="결과 마크다운 경로")
_ap.add_argument("--label", default="Evolved MoE", help="조건 이름(표 제목)")
_ap.add_argument("--max_new", type=int, default=None,
                 help="생성 토큰 수. 기본: MCQA 8, 그 외 128")
_A = _ap.parse_args()

sp = rc.spec(_A.dataset)
MCQA = sp.answer_letters is not None
O = rc.FEAT_DIR
BINNED = Path(_A.binned)
SRC = REPO / sp.src[sp.eval_split]
DENSE = Path(_A.dense)
BASE = sp.base_model
CKPT = Path(_A.ckpt) if _A.ckpt else REPO / sp.ckpt
OUT = Path(_A.out)
# MCQA는 답 글자 하나면 되지만 생성형 도메인은 더 필요하다.
BATCH, MAXLEN = 48, 1024
MAXNEW = _A.max_new if _A.max_new is not None else (8 if MCQA else 128)
DEV = "cuda"
np.random.seed(0)
torch.manual_seed(0)


# ---- 정답지 / 소스 ----
binned = [json.loads(l) for l in open(BINNED)]
bids = [str(r["id"]) for r in binned]
EX = sorted(binned[0]["per_expert"])
S = np.array([[r["per_expert"].get(e, 0) for e in EX] for r in binned], np.float32)
N, E = S.shape
src = {str(json.loads(l)["id"]): json.loads(l) for l in open(SRC, encoding="utf-8")}
best_single, oracle_union = rc.baselines(S)
dense_rows = [json.loads(l) for l in open(DENSE, encoding="utf-8")]
dense_acc = 100 * np.mean([float(r["pass_score"]) > 0 for r in dense_rows])


def align(fp, ip):
    """특징 행렬을 binned 순서(bids)에 맞춘다."""
    F = np.load(fp)
    idx = {p: i for i, p in enumerate(rc.load_ids(ip))}
    return np.array([F[idx[p]] for p in bids], np.float32)


# ---- feature ----
HS = align(rc.feat_path(sp, sp.eval_split, "hs_mean"), rc.feat_path(sp, sp.eval_split, "hs_ids"))
_emb_npy, _emb_ids = rc.emb_paths(sp, "eval")
EMB = align(_emb_npy, _emb_ids)
CONF = PRED = ANS = None
if MCQA:
    conf_npz = Path(_A.conf) if _A.conf else rc.feat_path(sp, sp.eval_split, "conf")
    cd = np.load(conf_npz, allow_pickle=True)
    ci = {p: i for i, p in enumerate(str(x) for x in cd["ids"])}
    cj = {e: j for j, e in enumerate(str(x) for x in cd["experts"])}
    CONF = np.array([[cd["conf"][ci[p], cj[e]] for e in EX] for p in bids], np.float32)
    PRED = np.array([[cd["pred"][ci[p], cj[e]] for e in EX] for p in bids], np.int64)
    ANS = align(rc.feat_path(sp, sp.eval_split, "ansprob"),
                rc.feat_path(sp, sp.eval_split, "ansprob_ids"))


def top2(score):
    return score.argsort(1)[:, -2:][:, ::-1]


def zc(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)


def rankc(X):    # 예전 이름은 rc — router_common 모듈명과 충돌해서 바꿨다
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
mlp_log = {"MLP hidden-state": cv_logits(HS), "MLP encoder-emb": cv_logits(EMB)}
if MCQA:
    mlp_log["MLP answer-prob"] = cv_logits(ANS)
    mlp_log["MLP confidence"] = cv_logits(CONF)
    mlp_log["MLP hs+conf"] = cv_logits(np.concatenate([HS, CONF], 1))

prior = S.mean(0, keepdims=True)
rng = np.random.default_rng(0)
rand2 = np.array([rng.choice(E, 2, replace=False) for _ in range(N)])

# ---- 방법 → top-2 (N,2) 픽 ----
# 도메인 무관 4종. 나머지 8종은 confidence/답분포에 의존해 MCQA에서만 붙는다.
methods = {
    "random-2": rand2,
    "MLP hidden-state": top2(mlp_log["MLP hidden-state"]),
    "MLP encoder-emb": top2(mlp_log["MLP encoder-emb"]),
}
if MCQA:
    # pred-agreement: 확신 가중 다수결 답에 동의하는 expert를 우대
    n_letters = len(sp.answer_letters)
    agree = np.zeros((N, E), np.float32)
    for i in range(N):
        v = np.zeros(n_letters, np.float32)
        for e in range(E):
            v[PRED[i, e]] += CONF[i, e]
        plur = v.argmax()
        for e in range(E):
            agree[i, e] = (1.0 if PRED[i, e] == plur else 0.0) + 0.01 * CONF[i, e]
    methods.update({
        "confidence (raw)": top2(CONF),
        "confidence (z-norm)": top2(zc(CONF)),
        "confidence (rank)": top2(rankc(CONF)),
        "confidence + prior": top2(zc(CONF) + 3.0 * zc(np.repeat(prior, N, 0))),
        "pred-agreement": top2(agree),
        "MLP answer-prob": top2(mlp_log["MLP answer-prob"]),
        "MLP confidence": top2(mlp_log["MLP confidence"]),
        "MLP hs+conf": top2(mlp_log["MLP hs+conf"]),
    })
# oracle은 tie-break에만 confidence를 쓰므로 없으면 solve 매트릭스만으로 정한다.
methods["oracle top-2"] = top2(S * 10 + (CONF if MCQA else 0.0))

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


_GEN_SYS, _GEN_USER = sp.gen


def prompt(i):
    r = src[i]
    msgs = [{"role": "system", "content": _GEN_SYS},
            {"role": "user", "content": _GEN_USER.format(instruction=r["instruction"])}]
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
    """원본 레코드를 그대로 넘겨 scoring_kind 디스패치를 탄다(도메인 무관)."""
    r = dict(src[bids[i]])
    r.setdefault("dataset", sp.name)
    r.setdefault("scoring_kind", sp.name)
    return 1 if score_one(r, text) > 0 else 0


# ---- 방법별 정확도 ----
results = {}
for m in methods:
    accs = [score_text(i, memo[(method_pair[m][i], i)]) for i in range(N)]
    results[m] = 100 * np.mean(accs)
    print(f"  [{m}] {results[m]:.1f}%", flush=True)

# ==== 출력 ====
order = [m for m in
         ["random-2", "confidence (raw)", "confidence (z-norm)", "confidence (rank)",
          "confidence + prior", "pred-agreement", "MLP hidden-state", "MLP encoder-emb",
          "MLP answer-prob", "MLP confidence", "MLP hs+conf", "oracle top-2"]
         if m in results]
lines = [
    f"# {sp.name.upper()} End-to-End 배포: {_A.label} (top-2 0.5 병합) vs Dense SFT",
    "",
    f"- **앵커 (Dense SFT baseline)**: **{dense_acc:.2f}%** ({DENSE.name})",
    f"- 조건: **{_A.label}** (experts={E}) · 참조: best-single(solve) {best_single:.1f}% · oracle-union {oracle_union:.1f}%",
    f"- 방식: 라우팅 방법이 고른 top-2 어댑터를 linear 0.5/0.5로 병합 → {N}문제 실제 생성 → EM 채점",
    "",
    f"| 라우팅 방법 | 배포 정확도(%) | vs Dense({dense_acc:.2f}) |",
    "|---|---|---|",
]
for m in order:
    lines.append(f"| {m} | {results[m]:.1f} | {results[m]-dense_acc:+.1f} |")
txt = "\n".join(lines) + "\n"
OUT.write_text(txt, encoding="utf-8")
print("\n" + txt)
print(f"saved -> {OUT}")

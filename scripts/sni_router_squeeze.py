#!/usr/bin/env python
"""라우팅 쥐어짜기 — 재생성 0. 네 가지를 한 번에 잰다.

1. 선택 규칙: k명의 실제 출력에서 정답 없이 하나 고르기(다수결 / 최단 / 최장 / 라우터 1등)
   ⚠️ 우리 축이 "짧은 닫힌 답 vs 긴 자유서술"이므로 최단 규칙이 후보다.
2. 선택적 개입: 라우터 로짓 마진(top1−top2) 상위 q%에서만 라우팅, 나머지는 best-single 고정.
3. 예산 곡선: B=1..5에서 라우터 top-B / 무작위 B명 / 한 사람 B회.
4. 일반화 갭: train 98 vs test 64 — 선형 + 강한 정규화로 외우기를 막으면 test가 오르나.

라우터는 `sni_router_train.py`에서 dev가 고른 설정(hs_last, head, 512/2/120/0.3/1e-3)을 재학습해
로짓을 얻는다. 학습 대상은 contested, 라벨은 p̂ 소프트.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import router_common as rc                       # noqa: E402
from evaluation.scorer import _sni_normalize     # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RAW = "results/sni/binning_seed20212003"


def mlp(d, hid, nl, drop, E):
    if nl == 0:
        return nn.Linear(d, E)
    L = [nn.Linear(d, hid), nn.ReLU(), nn.Dropout(drop)]
    for _ in range(nl - 2):
        L += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop)]
    return nn.Sequential(*L, nn.Linear(hid, E))


def train(Xf, Sf, Xs, cfg, seeds=(42, 1, 7)):
    hid, nl, ep, drop, wd = cfg
    out = []
    for s in seeds:
        torch.manual_seed(s)
        net = mlp(Xf.shape[1], hid, nl, drop, Sf.shape[1]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(ep):
            p = torch.randperm(len(Xf), device=DEV)
            for i in range(0, len(Xf), 256):
                b = p[i:i + 256]
                opt.zero_grad(); lf(net(Xf[b]), Sf[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out.append([net(x).cpu().numpy() for x in Xs])
    return [np.mean([o[i] for o in out], 0) for i in range(len(Xs))]


def load_raw(path, norm=True):
    """pid -> cid -> [(정규화답, 원문길이, pass)]"""
    c = defaultdict(lambda: defaultdict(list))
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t = r.get("code") or ""
        c[r["pid"]][r["cid"]].append((_sni_normalize(t) if norm else t, len(t.split()), int(r["pass"])))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sni/router_squeeze.md")
    a = ap.parse_args()
    rng = random.Random(0)
    sp = rc.spec("sni")
    trb, teb = rc.labels(sp)
    ex = rc.experts(sp)
    names = {p["id"]: p["name"] for p in json.load(open("results/sni/seed20212003/roster_final.json"))}

    ids_tr, Xtr, Str = rc.align(rc.feat_path(sp, "train", "hs_last"),
                                rc.feat_path(sp, "train", "hs_ids"), trb, ex)
    ids_te, Xte, Ste = rc.align(rc.feat_path(sp, "test", "hs_last"),
                                rc.feat_path(sp, "test", "hs_ids"), teb, ex)
    idx = np.random.default_rng(0).permutation(len(Xtr)); cut = int(len(idx) * .9)
    fi, di = idx[:cut], idx[cut:]
    h = (Str[fi] > .5).sum(1); fic = fi[(h > 0) & (h < len(ex))]
    mu, sd = Xtr[fic].mean(0, keepdims=True), Xtr[fic].std(0, keepdims=True) + 1e-6
    Z = lambda M: torch.tensor((M - mu) / sd, dtype=torch.float32).to(DEV)
    Xf, Xd, Xt = Z(Xtr[fic]), Z(Xtr[di]), Z(Xte)
    Sf = torch.tensor(Str[fic], dtype=torch.float32).to(DEV)
    Sd, N = Str[di], len(Ste)
    bj = int(Str[fi].mean(0).argmax())
    bs = 100 * Ste[:, bj].mean()
    orc = 100 * Ste.max(1).mean()
    L = ["# 라우팅 쥐어짜기 (진화 16명, 재생성 0)", "",
         f"- test {N:,} · best-single(train) **{bs:.2f}** ({names.get(ex[bj], ex[bj])}) · "
         f"문제별 오라클 top-1 {orc:.2f} · 폭 {orc-bs:.2f}pp", ""]

    # ---------- 4. 일반화 갭
    L += ["## 4. 일반화 갭 — 외우기를 막으면 오르나", "",
          "| 모델 | wd | train top-1 | dev top-1 | **test top-1** |", "|---|---:|---:|---:|---:|"]
    best_logit, best_dev, best_tag = None, -1, ""
    for tag, cfg in [("MLP 512×2 (기존 최적)", (512, 2, 120, .3, 1e-3)),
                     ("MLP 512×2 강한 wd", (512, 2, 120, .3, 1e-1)),
                     ("선형", (0, 0, 60, 0., 1e-2)),
                     ("선형 강한 wd", (0, 0, 60, 0., 1.0))]:
        lf_, ld, lt = train(Xf, Sf, [Xf[:20000], Xd, Xt], cfg)
        acc = lambda P, S: 100 * float(np.mean(S[np.arange(len(S)), P.argmax(1)]))
        dv = acc(ld, Sd)
        L.append(f"| {tag} | {cfg[4]:g} | {acc(lf_, Str[fic][:20000]):.2f} | {dv:.2f} | "
                 f"**{acc(lt, Ste):.2f}** |")
        if dv > best_dev:
            best_dev, best_logit, best_tag = dv, lt, tag
    L += ["", f"이하 라우터 = dev 최고인 `{best_tag}`", ""]

    # ---------- 2. 선택적 개입
    R = best_logit
    srt = np.sort(R, 1)
    margin = srt[:, -1] - srt[:, -2]
    order = np.argsort(-margin)
    L += ["## 2. 선택적 개입 — 확신할 때만 라우팅", "",
          "| 개입 비율 | test | best-single 대비 |", "|---:|---:|---:|"]
    for q in (10, 25, 50, 75, 100):
        k = int(N * q / 100)
        pick = np.full(N, bj); pick[order[:k]] = R.argmax(1)[order[:k]]
        v = 100 * float(np.mean(Ste[np.arange(N), pick]))
        L.append(f"| {q}% | {v:.2f} | {v-bs:+.2f} |")

    # ---------- 1·3. 실제 출력에서 고르기 + 예산 곡선
    te = load_raw(f"{RAW}/test_raw.jsonl")
    pids = ids_te
    rank = np.argsort(-R, 1)

    def choose(cands, rule):
        """cands = [(정규화답, 길이, pass)]"""
        if rule == "majority":
            c = Counter(x[0] for x in cands); top = max(c.values())
            best = [k for k, v in c.items() if v == top]
            p = rng.choice(best)
        elif rule == "shortest":
            p = min(cands, key=lambda x: (x[1], x[0]))[0]
        elif rule == "longest":
            p = max(cands, key=lambda x: (x[1], x[0]))[0]
        return next(x[2] for x in cands if x[0] == p)

    L += ["", "## 1·3. 예산 B회를 어떻게 쓰나 (정답 없이 고름)", "",
          "| B | 후보 | 다수결 | 최단 | 최장 | union(오라클) |", "|---:|---|---:|---:|---:|---:|"]
    for B in (2, 3, 5):
        for tag, grp_fn in (("라우터 top-B", lambda i, B=B: [ex[j] for j in rank[i, :B]]),
                            ("무작위 B명", lambda i, B=B: rng.sample(ex, B))):
            res = {r: 0 for r in ("majority", "shortest", "longest")}
            uni = 0
            for i, pid in enumerate(pids):
                g = grp_fn(i)
                cands = [te[pid][c][0] for c in g]
                for r in res:
                    res[r] += choose(cands, r)
                uni += any(x[2] for x in cands)
            L.append(f"| {B} | {tag} | {100*res['majority']/N:.2f} | {100*res['shortest']/N:.2f} | "
                     f"{100*res['longest']/N:.2f} | {100*uni/N:.2f} |")
        if B <= 3:   # 같은 사람 B회 (K=3까지)
            res = {r: 0 for r in ("majority", "shortest", "longest")}; uni = 0
            for pid in pids:
                cands = te[pid][ex[bj]][:B]
                for r in res:
                    res[r] += choose(cands, r)
                uni += any(x[2] for x in cands)
            L.append(f"| {B} | 같은 1명 × B회 | {100*res['majority']/N:.2f} | "
                     f"{100*res['shortest']/N:.2f} | {100*res['longest']/N:.2f} | {100*uni/N:.2f} |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

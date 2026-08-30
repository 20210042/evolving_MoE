#!/usr/bin/env python
"""분할 사다리를 **공식 지표(EM · ROUGE-L)**로 — test 8,699. 재생성 0.

`sni_partition_compare.py`는 우리 이진 통과율(EM==100 or ROUGE-L>70)로 잰다.
협업자 표(Dense 55.63/68.93, MoE 56.24/69.47)와 같은 통화로 놓으려면 EM·ROUGE 원값이 필요하다.
저장된 생성 문자열만 다시 잰다(재생성 0).

분할마다:
  · 그룹 앎(train)   — 그룹별 최적 expert를 **train에서** 고르고 test에서 평가
  · 분할 오라클(test) — 그룹별 최적을 **test에서** 고른다(라벨 해상도 천장)
지표마다 최적 expert가 다르므로 EM·ROUGE를 각각 따로 고른다.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import router_common as rc  # noqa: E402
from evaluation.scorer import sni_metrics  # noqa: E402

CACHE = Path("results/sni/official_matrices.npz")
REPORT = "results/sni/partition_compare_official.md"
RAW = {"train": "results/sni/binning_seed20212003/train_raw.jsonl",
       "test": "results/sni/binning_seed20212003/test_raw.jsonl"}
SRC = {"train": "export/sni_v4/sni_train.jsonl", "test": "export/sni_v4/sni_test.jsonl"}


def score_split(split, ex):
    src = {}
    for l in open(SRC[split], encoding="utf-8"):
        r = json.loads(l)
        src[r["id"]] = r
    pids = sorted(src)
    pi = {p: i for i, p in enumerate(pids)}
    ei = {c: i for i, c in enumerate(ex)}
    EM = np.zeros((len(pids), len(ex)), np.float32)
    RL = np.zeros((len(pids), len(ex)), np.float32)
    cache, n = {}, 0
    for l in open(RAW[split], encoding="utf-8"):
        d = json.loads(l)
        if int(d["rep"]) != 0 or d["cid"] not in ei or d["pid"] not in pi:
            continue                       # rep 0 = 단일 생성 (협업자 수치와 같은 층)
        key = (d["pid"], d.get("code") or "")
        m = cache.get(key)
        if m is None:
            m = cache[key] = sni_metrics(src[d["pid"]], key[1])
        EM[pi[d["pid"]], ei[d["cid"]]] = m["exact_match"]
        RL[pi[d["pid"]], ei[d["cid"]]] = m["rougeL"]
        n += 1
        if n % 100000 == 0:
            print(f"  {split} {n:,}", flush=True)
    return pids, EM, RL


def topk_partition(lab, k):
    u, c = np.unique(lab, return_counts=True)
    top = set(u[np.argsort(-c)[:k - 1]])
    return np.array([x if x in top else "__rest__" for x in lab])


def main():
    sp = rc.spec("sni")
    ex = rc.experts(sp)
    k = len(ex)
    rng = np.random.default_rng(0)
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        ids_tr, EMtr, RLtr = list(z["ids_tr"]), z["EMtr"], z["RLtr"]
        ids_te, EMte, RLte = list(z["ids_te"]), z["EMte"], z["RLte"]
    else:
        ids_tr, EMtr, RLtr = score_split("train", ex)
        ids_te, EMte, RLte = score_split("test", ex)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, ids_tr=np.array(ids_tr), EMtr=EMtr, RLtr=RLtr,
                            ids_te=np.array(ids_te), EMte=EMte, RLte=RLte)

    meta = {}
    for split in ("train", "test"):
        for l in open(SRC[split], encoding="utf-8"):
            r = json.loads(l)
            meta[r["id"]] = (r.get("category") or "?", r.get("sni_domain") or "?")
    cat = (np.array([meta[i][0] for i in ids_tr]), np.array([meta[i][0] for i in ids_te]))
    dom = (np.array([meta[i][1] for i in ids_tr]), np.array([meta[i][1] for i in ids_te]))

    parts = {}
    Dtr = EMtr - EMtr.mean(1, keepdims=True)
    Dte = EMte - EMte.mean(1, keepdims=True)
    parts["ours (진화 16명 배정)"] = ((Dtr + rng.random(Dtr.shape) * 1e-9).argmax(1),
                                      (Dte + rng.random(Dte.shape) * 1e-9).argmax(1))
    for tag, (ltr, lte) in (("category 상위", cat), ("domain 상위", dom)):
        ptr, pte = topk_partition(ltr, k), topk_partition(lte, k)
        u = {v: i for i, v in enumerate(sorted(set(ptr) | set(pte)))}
        parts[tag] = (np.array([u[x] for x in ptr]), np.array([u[x] for x in pte]))
    parts["random (균등)"] = (rng.integers(0, k, len(ids_tr)), rng.integers(0, k, len(ids_te)))
    sf = Path("export/sni_split_seed20212003/split.jsonl")
    if sf.exists():
        tot = np.zeros(k)
        for line in open(sf, encoding="utf-8"):
            d = json.loads(line)
            for c in d["experts"]:
                if c in ex:
                    tot[ex.index(c)] += 1
        pr = tot / tot.sum()
        parts["random (우리 분할 크기 맞춤)"] = (rng.choice(k, len(ids_tr), p=pr),
                                                 rng.choice(k, len(ids_te), p=pr))

    def ladder(gtr, gte, Mtr, Mte):
        pick = {j: int(Mtr[gtr == j].mean(0).argmax()) for j in np.unique(gtr)}
        known = float(np.mean([Mte[i, pick.get(gte[i], 0)] for i in range(len(Mte))]))
        pick_te = {j: int(Mte[gte == j].mean(0).argmax()) for j in np.unique(gte)}
        orac = float(np.mean([Mte[i, pick_te[gte[i]]] for i in range(len(Mte))]))
        return known, orac

    bs_em = EMte[:, int(EMtr.mean(0).argmax())].mean()
    bs_rl = RLte[:, int(RLtr.mean(0).argmax())].mean()
    L = ["# 분할 사다리 — 공식 지표(EM · ROUGE-L), test 8,699 (재생성 0)", "",
         "rep 0 단일 생성. 협업자 표(Dense 55.63/68.93 · 4x1 MoE 56.24/69.47)와 같은 통화다.",
         f"- best-single(train 선택): EM **{bs_em:.2f}** · ROUGE-L **{bs_rl:.2f}**",
         f"- 문제별 오라클(16명 중 최선): EM **{EMte.max(1).mean():.2f}** · "
         f"ROUGE-L **{RLte.max(1).mean():.2f}**",
         f"- 16명 평균(무작위 배정): EM {EMte.mean():.2f} · ROUGE-L {RLte.mean():.2f}", "",
         "| 분할 (k=16) | EM 그룹앎(train) | EM 분할오라클 | ROUGE 그룹앎(train) | ROUGE 분할오라클 |",
         "|---|---:|---:|---:|---:|"]
    for tag, (gtr, gte) in parts.items():
        ke, oe = ladder(gtr, gte, EMtr, EMte)
        kr, orr = ladder(gtr, gte, RLtr, RLte)
        L.append(f"| {tag} | {ke:.2f} | **{oe:.2f}** | {kr:.2f} | **{orr:.2f}** |")
    L += ["", "⚠️ `ours` 행의 두 칸은 분할이 문제 단위로 정의돼 항등식이다(문제별 오라클과 같다). "
              "값어치를 재는 칸이 아니라 **분할이 담은 상한**을 보이는 칸이다.", ""]
    Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

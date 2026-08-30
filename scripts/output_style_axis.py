#!/usr/bin/env python3
"""명제 A: persona가 출력분포를 계통적으로 옮기는가 (정답·실패 축과 무관하게).

failure_mode_by_expert.py와 **같은 검정 틀**을 쓴다 — 문제 내에서 expert 라벨을 셔플해
귀무분포를 만든다(문제 고유 성향 보존, expert 귀속만 파괴). 그래야 p값이 직접 비교된다.

측정: 생성 코드의 스타일 특징(길이·주석·제어구조·식별자 등)을 문제 단위로 중심화한 뒤,
expert 주효과 크기 = Σ_j (expert j의 중심화 평균)² 를 귀무와 비교한다.
추가로 문제 내 12개 출력의 pairwise 토큰 Jaccard 평균(= 출력 다양성의 절대 크기)을 낸다.

Usage:
  python scripts/output_style_axis.py --input results/acc/seed20211004/binning_test_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

TOK = re.compile(r"[A-Za-z_]\w*|\d+|[^\sA-Za-z_\d]")
KEYWORDS = ("for", "while", "if", "else", "elif", "def", "class", "try", "except",
            "return", "import", "lambda", "yield", "with", "assert")


def features(code: str) -> dict:
    lines = code.splitlines()
    toks = TOK.findall(code)
    n_tok = max(len(toks), 1)
    comment = sum(1 for l in lines if l.strip().startswith("#"))
    idents = [t for t in toks if re.fullmatch(r"[A-Za-z_]\w*", t)]
    f = {
        "n_chars": len(code),
        "n_lines": len(lines),
        "n_tokens": len(toks),
        "comment_ratio": comment / max(len(lines), 1),
        "mean_line_len": np.mean([len(l) for l in lines]) if lines else 0.0,
        "ident_len": np.mean([len(t) for t in idents]) if idents else 0.0,
        "uniq_ident_ratio": len(set(idents)) / max(len(idents), 1),
        "digit_ratio": sum(1 for t in toks if t.isdigit()) / n_tok,
    }
    for k in KEYWORDS:
        f[f"kw_{k}"] = sum(1 for t in toks if t == k) / n_tok
    return f


def token_set(code: str) -> set:
    return set(TOK.findall(code))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/acc/seed20211004/binning_test_full.jsonl")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / a.input, encoding="utf-8") if l.strip()]
    experts: list[str] = []
    for r in rows:
        for e in (r.get("expert_outputs") or {}):
            if e not in experts:
                experts.append(e)
    E = len(experts)
    keep = [r for r in rows if len(r.get("expert_outputs") or {}) == E]
    N = len(keep)

    names = list(features("x = 1").keys())
    X = np.zeros((N, E, len(names)))
    jac = []
    for i, r in enumerate(keep):
        outs = r["expert_outputs"]
        for j, e in enumerate(experts):
            X[i, j] = [features(outs[e])[k] for k in names]
        sets = [token_set(outs[e]) for e in experts]
        pw = [len(sets[u] & sets[v]) / max(len(sets[u] | sets[v]), 1)
              for u in range(E) for v in range(u + 1, E)]
        jac.append(float(np.mean(pw)))

    # 문제 단위 중심화 + 특징별 표준화
    Xc = X - X.mean(1, keepdims=True)
    sd = Xc.reshape(-1, len(names)).std(0) + 1e-12
    Xc /= sd

    def effect(mat: np.ndarray) -> np.ndarray:
        """expert별 중심화 평균의 제곱합 (특징별)."""
        return (mat.mean(0) ** 2).sum(0)          # (n_feat,)

    obs = effect(Xc)
    obs_tot = float(obs.sum())

    rng = np.random.default_rng(0)
    null = np.zeros((a.n_perm, len(names)))
    for t in range(a.n_perm):
        idx = np.argsort(rng.random((N, E)), axis=1)
        Xp = np.take_along_axis(Xc, idx[:, :, None], axis=1)
        null[t] = effect(Xp)
    null_tot = null.sum(1)

    def pval(o, nl):
        return float((np.sum(nl >= o) + 1) / (len(nl) + 1))

    order = np.argsort(-(obs - null.mean(0)) / (null.std(0) + 1e-12))
    L = [f"# 출력 스타일에 expert 주효과가 있는가 (명제 A) — `{a.input}`", "",
         f"- 문제 {N:,} × expert {E} · 특징 {len(names)}종 · 문제 단위 중심화",
         f"- 귀무: 문제 내 expert 라벨 셔플 {a.n_perm}회 (실패유형 검정과 동일 틀)",
         f"- 문제 내 12개 출력의 평균 토큰 Jaccard = **{np.mean(jac):.3f}** "
         f"(1.0이면 완전 동일, 낮을수록 서로 다른 코드)", "",
         f"**전체 expert 주효과: 관측 {obs_tot:.2f} vs 귀무 {null_tot.mean():.2f} ± {null_tot.std():.2f} "
         f"→ p = {pval(obs_tot, null_tot):.4f}**", "",
         "| 특징 | 관측 | 귀무 평균 | z | p |", "|---|---:|---:|---:|---:|"]
    for k in order[:12]:
        z = (obs[k] - null[:, k].mean()) / (null[:, k].std() + 1e-12)
        L.append(f"| `{names[k]}` | {obs[k]:.3f} | {null[:, k].mean():.3f} | {z:+.2f} | "
                 f"{pval(obs[k], null[:, k]):.4f} |")
    L += ["", "읽는 법: p가 유의하면 persona는 출력분포를 계통적으로 옮긴다(축은 실재). 그 축이",
          "정답·실패 축과 직교하는지는 failure_modes / binning 결과와 대조해서 판단한다.", ""]

    out = ROOT / (a.out or (str(Path(a.input).with_suffix("")) + ".style_axis.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

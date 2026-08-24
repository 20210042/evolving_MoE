#!/usr/bin/env python3
"""⚠️ 폐기(2026-08-24). 이 스크립트는 내가 만든 형식 제거 규칙(_sni_extract)의 완화 폭을
재던 도구인데, 그 규칙 자체가 공식 채점에 없는 자체 발명이라 삭제됐다.
SNI 채점은 공식(eval/automatic/evaluation.py) EM·ROUGE-L로 되돌렸다 —
src/evaluation/scorer.py의 sni_metrics를 볼 것. 이 파일은 실행되지 않는다.
"""
raise SystemExit(__doc__)

# --- 이하 원본 보존 ---
# #!/usr/bin/env python3
# """SNI 채점 추출규칙의 **완화 폭**을 기존 생성물로 먼저 재는 감사 도구.
# 
# 프로브 job 229352의 생성물(results/sni/probe_raw.jsonl, 41,400건)이 그대로 남아 있으므로
# GPU 없이 규칙 전/후 EM을 재계산할 수 있다. "고쳤다"가 아니라 "얼마나 올랐고 그게 정당한가"를
# 숫자와 표본으로 먼저 보고하기 위한 것이다(docs/PLAN_sni_probe_v2.md §4).
# 
#   전(before) : 정규화 후 완전일치        = 원 채점
#   후(after)  : _sni_extract 적용 후 일치 = 사전등록된 형식 제거 4종
# 
# Usage:
#   python scripts/sni_rescore_audit.py \
#       --raw results/sni/probe_raw.jsonl --data export/sni_v2/sni_all.jsonl \
#       --out results/sni/scorer_extract_audit.md
# """
# from __future__ import annotations
# 
# import argparse
# import collections
# import json
# import random
# import sys
# from pathlib import Path
# 
# ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(ROOT / "src"))
# 
# from evaluation.scorer import (  # noqa: E402
#     _sni_extract, _sni_normalize, _sni_refs, score_sni_item,
# )
# 
# 
# def em_before(item: dict, pred: str) -> float:
#     """규칙 적용 전 = 원 채점(정규화 후 완전일치)."""
#     p = _sni_normalize(pred)
#     if not p:
#         return 0.0
#     return 100.0 if any(p == _sni_normalize(r) for r in _sni_refs(item)) else 0.0
# 
# 
# def em_after(item: dict, pred: str) -> float:
#     return score_sni_item(item, pred)
# 
# 
# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--raw", default="results/sni/probe_raw.jsonl")
#     ap.add_argument("--data", default="export/sni_v2/sni_all.jsonl")
#     ap.add_argument("--out", default="results/sni/scorer_extract_audit.md")
#     ap.add_argument("--n_sample", type=int, default=100, help="수동검수용 표본 수")
#     ap.add_argument("--seed", type=int, default=0)
#     a = ap.parse_args()
# 
#     items = {}
#     with open(ROOT / a.data, encoding="utf-8") as f:
#         for line in f:
#             d = json.loads(line)
#             items[d["id"]] = d
# 
#     tot = 0
#     gained: list = []        # 0 → 100
#     lost: list = []          # 100 → 0  (있으면 규칙이 잘못된 것)
#     by_kind = collections.Counter()
#     n_by_kind = collections.Counter()
#     with open(ROOT / a.raw, encoding="utf-8") as f:
#         for line in f:
#             d = json.loads(line)
#             it = items.get(d["pid"])
#             if it is None:
#                 continue
#             tot += 1
#             kind = "closed" if it.get("task_closed") else "open"
#             n_by_kind[kind] += 1
#             b, af = em_before(it, d["code"]), em_after(it, d["code"])
#             if b == af:
#                 continue
#             rec = {"pid": d["pid"], "cid": d["cid"], "kind": kind,
#                    "task": it["task_name"], "gt": _sni_refs(it)[:3],
#                    "out": d["code"], "extracted": _sni_extract(it, d["code"])}
#             if af > b:
#                 gained.append(rec)
#                 by_kind[kind] += 1
#             else:
#                 lost.append(rec)
# 
#     rng = random.Random(a.seed)
#     sample = rng.sample(gained, min(a.n_sample, len(gained))) if gained else []
# 
#     L = ["# SNI 채점 추출규칙 — 완화 폭 감사", "",
#          f"- 대상: `{a.raw}` {tot:,}건 (job 229352 생성물, 재생성 없음)",
#          f"- **0점 → 100점 {len(gained):,}건 ({100*len(gained)/max(1,tot):.2f}%)**",
#          f"- 100점 → 0점 {len(lost):,}건  ← 0이 아니면 규칙이 틀린 것", "",
#          "| 층 | 생성 건수 | 점수 오른 건 | 비율 |", "|---|---:|---:|---:|"]
#     for k in ("closed", "open"):
#         n = n_by_kind[k]
#         L.append(f"| {k} | {n:,} | {by_kind[k]:,} | {100*by_kind[k]/max(1,n):.2f}% |")
#     L += ["", "---", "",
#           f"## 수동검수 표본 {len(sample)}건 — 오른 것이 **정말 정답인가**", "",
#           "판정: 정답이면 O, 형식만 맞고 내용이 틀렸으면 X. X가 나오면 해당 규칙을 되돌린다.", "",
#           "| # | 층 | gold | 모델 출력 | 추출 결과 | 판정 |", "|---:|---|---|---|---|:--:|"]
#     for i, r in enumerate(sample, 1):
#         gt = " / ".join(x.replace("|", "\\|")[:40] for x in r["gt"])
#         out = r["out"].replace("|", "\\|").replace("\n", " ⏎ ")[:90]
#         ex = r["extracted"].replace("|", "\\|")[:50]
#         L.append(f"| {i} | {r['kind'][:1]} | `{gt}` | `{out}` | `{ex}` |  |")
#     if lost:
#         L += ["", "## ⚠️ 점수가 내려간 건 (규칙 결함)", ""]
#         for r in lost[:30]:
#             L.append(f"- `{r['task']}` gold=`{r['gt'][0][:40]}` out=`{r['out'][:60]}` "
#                      f"→ 추출 `{r['extracted'][:40]}`")
#     out_path = ROOT / a.out
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
#     print(f"총 {tot:,} · 상승 {len(gained):,} · 하락 {len(lost):,} -> {out_path}")
# 
# 
# if __name__ == "__main__":
#     main()

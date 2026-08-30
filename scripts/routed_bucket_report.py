#!/usr/bin/env python3
"""LLM 라우팅(top-1) 결과를 버킷별로 집계 — 보고용 한 줄 표.

버킷은 같은 로스터의 test binning 매트릭스로 정의한다:
  All Failed  : 전원 실패 (어떤 선택도 0점)
  All Passed  : 전원 성공 (어떤 선택도 100점)
  w/o All Failed = 전체 − All Failed
score_outputs.py는 집계만 저장하므로 여기서 문항별로 다시 채점한다(실행 채점, GPU 불필요).

Usage:
  python scripts/routed_bucket_report.py \
      --input results/acc/seed20211004/inference_test_routed_20211004_greedy.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.loader import get_dataset  # noqa: E402
from orchestrator import _score_pair  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--binned", default="results/acc/seed20211004/binning_test_full.binned.jsonl")
    ap.add_argument("--dataset", default="acc")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default="export/acc_v2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--label", default="Evolved Roster (LLM top-1 routing)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(ROOT / a.input, encoding="utf-8") if l.strip()]
    items = {str(it["id"]): it for it in get_dataset(a.dataset, split=a.split, local_dir=a.data_dir)}
    binned = {str(json.loads(l)["id"]): json.loads(l)
              for l in open(ROOT / a.binned, encoding="utf-8") if l.strip()}
    E = len(next(iter(binned.values()))["per_expert"])

    args = []
    for r in recs:
        pid = str(r.get("id"))
        code = r.get("final_code") or r.get("final_output") or r.get("code") or r.get("prediction", "")
        if pid in items:
            args.append((items[pid], code, 10, 3.0, "release_v5", pid))
    scores: dict = {}
    pool = ProcessPoolExecutor(max_workers=a.workers)
    futs = {pool.submit(_score_pair, x): x[5] for x in args}
    try:
        for f in as_completed(futs, timeout=max(1800.0, 2.0 * len(args))):
            _pid, key, sc = f.result()
            scores[key] = sc
    except FuturesTimeoutError:
        print(f"⚠️ wall-clock timeout: {sum(1 for f in futs if not f.done())} 미완료 → 0.0")
    finally:
        for p in list(getattr(pool, "_processes", {}).values()):
            if p.is_alive():
                p.kill()
        pool.shutdown(wait=False, cancel_futures=True)
    for key in futs.values():
        scores.setdefault(key, 0.0)

    ok = {pid: (sc >= 100.0 - 1e-6) for pid, sc in scores.items()}
    n_of = {pid: sum(binned[pid]["per_expert"].values()) for pid in ok if pid in binned}

    def agg(sel):
        ids = [p for p in ok if p in n_of and sel(n_of[p])]
        return len(ids), (100.0 * sum(ok[p] for p in ids) / len(ids) if ids else float("nan"))

    n_all, a_all = agg(lambda n: True)
    n_af, a_af = agg(lambda n: n == 0)
    n_ap, a_ap = agg(lambda n: n == E)
    n_wo, a_wo = agg(lambda n: n > 0)
    n_ct, a_ct = agg(lambda n: 0 < n < E)

    L = [f"# {a.label} — 버킷별 성능", "",
         f"- 입력: `{a.input}` · 버킷 정의: `{a.binned}` (expert {E}명)", "",
         f"| Method | Overall [{n_all}] | w/o All Failed [{n_wo}] | All Failed [{n_af}] | All Passed [{n_ap}] |",
         "|---|---:|---:|---:|---:|",
         f"| {a.label} | {a_all:.2f} | {a_wo:.2f} | {a_af:.2f} | {a_ap:.2f} |", "",
         f"(참고: 정오답이 갈리는 Contested [{n_ct}]에서 {a_ct:.2f}. All Failed는 정의상 0, "
         f"All Passed는 100이므로 방법 간 차이는 이 구간에서만 생긴다.)", ""]

    out = ROOT / (a.out or (str(Path(a.input).with_suffix("")) + ".bucket_report.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

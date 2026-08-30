#!/usr/bin/env python3
"""persona가 정오답은 못 바꿔도 '실패의 종류'는 바꾸는가 — 기존 생성물 재채점(GPU 불필요).

binning은 채점기의 `status`(accepted/wrong_answer/runtime_error/timeout)를 버리고 0/1만
남긴다. 여기서는 같은 생성물을 다시 실행해 status를 살려 expert × 실패유형 분할표를 만든다.

가설: persona는 출력분포를 옮기지만 그 축이 정답축과 직교한다 → pass율은 같아도(로스터 폭
1.2pp) 실패유형 구성은 유의하게 갈려야 한다. 갈리지 않으면 persona는 실패의 성격조차 안 바꾼다.

검정: expert × status 분할표의 χ² 통계량을, **문제 내에서 expert 라벨을 셔플한** 귀무분포와
비교한다(문제 난이도·문제별 실패 성향을 보존하고 expert 귀속만 파괴). scipy 불필요.

Usage:
  python scripts/failure_mode_by_expert.py \
      --input results/acc/seed20211004/binning_test_full.jsonl \
      --dataset acc --split test --data_dir export/acc_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STATUSES = ["accepted", "wrong_answer", "runtime_error", "timeout", "other"]


def _run_one(args):
    """(item, code, key) -> (key, status, n_pass, n_total). 최상위 함수여야 pickle 된다."""
    item, code, key = args
    from evaluation.acc_exec import ExecutionInterface

    def _json(x):
        if not isinstance(x, str):
            return x
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x

    problem = dict(item)
    problem["eval_spec"] = _json(item.get("eval_spec")) or {}
    problem["test_cases"] = _json(item.get("test_cases")) or []
    try:
        out = ExecutionInterface().run(problem, code, solution_id="candidate")
    except Exception:
        return key, "other", 0, 0
    st = str(out.get("status") or "other")
    if st not in STATUSES:
        st = "other"
    return key, st, int(out.get("num_tests_passed") or 0), int(out.get("num_tests_total") or 0)


def chi2_stat(table: np.ndarray) -> float:
    row, col = table.sum(1, keepdims=True), table.sum(0, keepdims=True)
    tot = table.sum()
    if tot == 0:
        return 0.0
    exp = row @ col / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(exp > 0, (table - exp) ** 2 / exp, 0.0)
    return float(t.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/acc/seed20211004/binning_test_full.jsonl")
    ap.add_argument("--dataset", default="acc")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default="export/acc_v2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)

    records = [json.loads(l) for l in open(ROOT / a.input, encoding="utf-8") if l.strip()]
    logging.info("Loaded %d binning records.", len(records))

    from data.loader import get_dataset
    items = get_dataset(a.dataset, split=a.split, local_dir=a.data_dir)
    ref = {str(it["id"]): it for it in items}
    logging.info("Loaded %d reference items.", len(ref))

    experts: list[str] = []
    for r in records:
        for e in (r.get("expert_outputs") or {}):
            if e not in experts:
                experts.append(e)
    E = len(experts)

    res: dict = {}
    for start in range(0, len(records), a.chunk):
        chunk = records[start:start + a.chunk]
        args = []
        for r in chunk:
            pid = str(r.get("id"))
            item = ref.get(pid)
            if item is None:
                continue
            for e, code in (r.get("expert_outputs") or {}).items():
                args.append((item, code, (pid, e)))
        pool = ProcessPoolExecutor(max_workers=a.workers)
        futs = {pool.submit(_run_one, x): x[2] for x in args}
        try:
            for f in as_completed(futs, timeout=max(600.0, 2.0 * len(args) / a.workers)):
                key, st, np_, nt = f.result()
                res[key] = (st, np_, nt)
        except FuturesTimeoutError:
            logging.error("wall-clock timeout: %d unfinished -> other",
                          sum(1 for f in futs if not f.done()))
        finally:
            for p in list(getattr(pool, "_processes", {}).values()):
                if p.is_alive():
                    p.kill()
            pool.shutdown(wait=False, cancel_futures=True)
        for key in futs.values():
            res.setdefault(key, ("other", 0, 0))
        logging.info("Scored %d/%d problems.", min(start + a.chunk, len(records)), len(records))

    # expert × status 분할표
    idx = {e: j for j, e in enumerate(experts)}
    sidx = {s: k for k, s in enumerate(STATUSES)}
    table = np.zeros((E, len(STATUSES)), float)
    per_problem: dict[str, list[str]] = defaultdict(list)
    for (pid, e), (st, _p, _t) in res.items():
        table[idx[e], sidx[st]] += 1
        per_problem[pid].append(st)

    fail_cols = [sidx[s] for s in ("wrong_answer", "runtime_error", "timeout", "other")]
    obs_all = chi2_stat(table)
    obs_fail = chi2_stat(table[:, fail_cols])

    # 귀무: 문제 내에서 status를 expert에 무작위 재배정(문제별 결과 구성 보존)
    rng = np.random.default_rng(0)
    pids = list(per_problem)
    null_all, null_fail = np.zeros(a.n_perm), np.zeros(a.n_perm)
    for t in range(a.n_perm):
        tb = np.zeros_like(table)
        for pid in pids:
            sts = per_problem[pid]
            perm = rng.permutation(len(sts))
            for j, k in enumerate(perm):
                tb[j % E, sidx[sts[k]]] += 1
        null_all[t] = chi2_stat(tb)
        null_fail[t] = chi2_stat(tb[:, fail_cols])

    def pval(obs, null):
        return float((np.sum(null >= obs) + 1) / (len(null) + 1))

    L = [f"# expert별 실패유형 분포 — `{a.input}`", "",
         f"- 문제 {len(records):,} × expert {E} = {len(res):,}셀 재채점(GPU 불필요, 기존 생성물 재실행)",
         f"- 귀무: 문제 내 status 재배정 {a.n_perm}회 (문제별 결과 구성 보존, expert 귀속만 파괴)", "",
         "| expert | " + " | ".join(STATUSES) + " | pass율 |",
         "|---" * (len(STATUSES) + 2) + "|"]
    for e in experts:
        row = table[idx[e]]
        tot = row.sum()
        L.append(f"| `{e}` | " + " | ".join(f"{int(v)} ({100*v/tot:.1f}%)" for v in row) +
                 f" | {100*row[sidx['accepted']]/tot:.2f}% |")
    L += ["",
          f"χ²(expert × status 전체) 관측 **{obs_all:.1f}** vs 귀무 {null_all.mean():.1f} ± {null_all.std():.1f} "
          f"→ p = **{pval(obs_all, null_all):.4f}**",
          f"χ²(실패 유형만, accepted 제외) 관측 **{obs_fail:.1f}** vs 귀무 {null_fail.mean():.1f} ± {null_fail.std():.1f} "
          f"→ p = **{pval(obs_fail, null_fail):.4f}**", "",
          "읽는 법: pass율이 평평한데 실패유형 χ²가 유의하면 — persona는 출력분포를 옮기지만",
          "그 축이 정답축과 직교한다는 직접 증거다. 둘 다 무의미하면 persona는 실패의 성격조차 안 바꾼다.", ""]

    out = ROOT / (a.out or (str(Path(a.input).with_suffix("")) + ".failure_modes.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    cells = {f"{pid}\t{e}": {"status": st, "n_pass": p, "n_total": t} for (pid, e), (st, p, t) in res.items()}
    json.dump(cells, open(str(out).replace(".md", ".cells.json"), "w"))
    print("\n".join(L))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

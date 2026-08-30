#!/usr/bin/env python
"""학습 타깃을 gold로 할 때와 페르소나 출력으로 할 때 실제로 몇 %가 다른가 — 재생성 0.

`score_sni_item` = EM==100 **또는** ROUGE-L>70. 그래서 '맞힌 출력' 안에도 두 종류가 있다:
  · EM 통과   → 정규화하면 gold와 같다 = 타깃이 사실상 동일
  · ROUGE 통과 → gold와 다른 문자열이다 = 페르소나 정책이 남아 있는 몫
이 비율이 "골드나 페르소나나 똑같은가"의 답이다.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.scorer import _sni_normalize, _sni_refs, sni_metrics  # noqa: E402

gold = {}
for l in open("export/sni_v4/sni_test.jsonl", encoding="utf-8"):
    r = json.loads(l)
    gold[r["id"]] = r
n_pass = em = rouge_only = byte_diff = 0
per_expert = defaultdict(lambda: [0, 0])
var = defaultdict(set)
for line in open("results/sni/binning_seed20212003/test_raw.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if not r["pass"]:
        continue
    item = gold.get(r["pid"])
    if item is None:
        continue
    n_pass += 1
    pred = r.get("code") or ""
    npred = _sni_normalize(pred)
    refs = [_sni_normalize(x) for x in _sni_refs(item)]
    if npred in refs:
        em += 1
        per_expert[r["cid"]][0] += 1
        if pred.strip() not in [str(x).strip() for x in _sni_refs(item)]:
            byte_diff += 1
    else:
        rouge_only += 1
        per_expert[r["cid"]][1] += 1
    var[r["pid"]].add(npred)
print(f"통과 행 {n_pass:,}")
print(f"  EM 통과(정규화 후 gold와 동일) : {em:,} ({100*em/n_pass:.2f}%)")
print(f"     └ 그중 원문 문자열은 gold와 다름(대소문자·구두점·공백): {byte_diff:,} ({100*byte_diff/em:.2f}%)")
print(f"  ROUGE 통과(gold와 다른 문자열): {rouge_only:,} ({100*rouge_only/n_pass:.2f}%)")
mv = [len(v) for v in var.values() if v]
print(f"문제당 서로 다른 '맞힌 답' 종수: 중앙값 {sorted(mv)[len(mv)//2]} · 평균 {sum(mv)/len(mv):.2f} · 2종 이상 {100*sum(1 for x in mv if x>1)/len(mv):.1f}%")
print("\nexpert별 ROUGE 통과 비중(= 페르소나 문체가 남는 몫):")
for c, (e, ro) in sorted(per_expert.items(), key=lambda x: -x[1][1]/max(1, sum(x[1])))[:6]:
    print(f"  {c}: {100*ro/(e+ro):.1f}%  (EM {e:,} / ROUGE {ro:,})")

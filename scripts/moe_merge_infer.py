#!/usr/bin/env python3
"""Phase 3: Mixture-of-LoRA — top-2 어댑터 0.5/0.5 병합 → 단일 생성 → 채점.

각 val 문제에서 confidence 상위 2 expert를 add_weighted_adapter(linear,[0.5,0.5])로
병합해 단일 어댑터로 만든 뒤 '한 번' 생성해 최종 답. best-single(단일 top-1 생성)과
top-2 union UB(참조 상한)와 함께 배포 정확도 보고.

Usage:
  python scripts/moe_merge_infer.py [--weights 0.5,0.5] [--combo linear|cat|svd]
                                    [--route confidence|oracle] [--limit N]
검증: --weights 1,0 → 병합이 top-1 expert 단일 pass를 재현(배선 앵커).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
sys.path.insert(0, str(REPO / "src"))
from prompts import baseline_prompts as bp  # noqa: E402
from evaluation.scorer import score_qasc_item  # noqa: E402

CONF = REPO / "results/embed_viz_test/qasc_validation_lora_conf.npz"
BINNED = REPO / "results/qasc/seed20210211/inference_validation_lora13.binned.jsonl"
SRC = REPO / "export/qasc/qasc_validation.jsonl"
BASE = "meta-llama/Llama-3.1-8B-Instruct"
CKPT = REPO / "checkpoints/expert_sft/qasc_seed20210211_cap10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2, help="병합할 expert 수 (top-k)")
    ap.add_argument("--weights", default="", help="'0.5,0.5' 등. 생략시 균등(1/k) / 'conf'면 confidence 비례")
    ap.add_argument("--combo", default="linear", choices=["linear", "cat", "svd"])
    ap.add_argument("--route", default="confidence", choices=["confidence", "oracle"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    d = np.load(CONF, allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    experts = [str(x) for x in d["experts"]]
    conf = d["conf"].astype(np.float32)
    if a.limit:
        ids, conf = ids[:a.limit], conf[:a.limit]
    src = {str(json.loads(l)["id"]): json.loads(l) for l in open(SRC, encoding="utf-8")}
    binned = {str(json.loads(l)["id"]): json.loads(l) for l in open(BINNED, encoding="utf-8")} \
        if BINNED.is_file() else {}

    # 라우팅: 문제별 top-k expert index. confidence 직접 = raw self-confidence 상위 k.
    if a.route == "confidence":
        score_mat = conf
    else:  # oracle: real 매트릭스에서 그 문제를 푸는(=커버) 확신 상위 k (병합 상한용)
        S = np.array([[binned[i]["per_expert"].get(e, 0) for e in experts] for i in ids], np.float32)
        score_mat = S * 10 + conf  # 푸는 expert 우선, 동률은 확신
    topk = score_mat.argsort(1)[:, -a.k:][:, ::-1]                    # (N,k) 내림차순
    pick = {ids[i]: tuple(int(x) for x in topk[i]) for i in range(len(ids))}
    # 가중치: 균등(1/k) 기본 / 'conf'면 confidence 비례 / 명시값
    if a.weights == "conf":
        wmode = "conf"
    elif a.weights:
        fixed_w = [float(x) for x in a.weights.split(",")]
        wmode = "fixed"
    else:
        fixed_w = [1.0 / a.k] * a.k
        wmode = "fixed"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()
    model = PeftModel.from_pretrained(base, str(CKPT / experts[0]), adapter_name=experts[0])
    for e in experts[1:]:
        model.load_adapter(str(CKPT / e), adapter_name=e)
    model.eval()

    def prompt(i):
        r = src[i]
        msgs = [{"role": "system", "content": bp.QASC_GEN_SYSTEM},
                {"role": "user", "content": bp.QASC_GEN_USER.format(instruction=r["instruction"])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def gen_batch(idlist):
        outs = {}
        for s in range(0, len(idlist), a.batch):
            chunk = idlist[s:s + a.batch]
            enc = tok([prompt(i) for i in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len).to("cuda")
            with torch.no_grad():
                seq = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(seq[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for i, t in zip(chunk, dec):
                outs[i] = t.strip()
        return outs

    def score(i, text):
        r = src[i]
        item = {"id": i, "dataset": "qasc", "ground_truth": r["ground_truth"],
                "scoring_kind": "qasc", "instruction": r.get("instruction", "")}
        return 1 if score_qasc_item(item, text) > 0 else 0

    # (1) top-1 confidence 단일 생성 baseline (문제별 top-1 expert)
    by_e1 = defaultdict(list)
    for i in ids:
        by_e1[pick[i][0]].append(i)
    single = {}
    for ei, idlist in by_e1.items():
        model.set_adapter(experts[ei])
        single.update(gen_batch(idlist))
    single_acc = 100 * np.mean([score(i, single[i]) for i in ids])

    # (2) top-k 병합 단일 생성 (문제별 조합 그룹화 + 병합 캐싱)
    conf_by_id = {ids[t]: conf[t] for t in range(len(ids))}
    by_combo = defaultdict(list)
    for i in ids:
        by_combo[tuple(sorted(set(pick[i])))].append(i)
    merged = {}
    for combo, idlist in by_combo.items():
        combo = list(combo)
        if len(combo) == 1:
            model.set_adapter(experts[combo[0]])
        else:
            names = [experts[e] for e in combo]
            if wmode == "conf":
                # confidence 비례(조합 내 상대) — 그룹 평균 conf 사용
                cs = np.array([np.mean([conf_by_id[i][e] for i in idlist]) for e in combo])
                w = (cs / cs.sum()).tolist()
                aname = f"m__{'__'.join(names)}__confw"
            else:
                w = fixed_w[:len(combo)]
                w = [x / sum(w) for x in w] if a.combo == "linear" else w
                aname = f"m__{'__'.join(names)}__{a.combo}"
            if aname not in model.peft_config:
                model.add_weighted_adapter(names, w, adapter_name=aname, combination_type=a.combo)
            model.set_adapter(aname)
        merged.update(gen_batch(idlist))
    merged_acc = 100 * np.mean([score(i, merged[i]) for i in ids])

    # 참조 지표
    refs = {}
    if binned:
        S = np.array([[binned[i]["per_expert"].get(e, 0) for e in experts] for i in ids], np.float32)
        refs["best_single_real"] = 100 * S.mean(0).max()
        refs["oracle_union13"] = 100 * (S.sum(1) > 0).mean()
        refs[f"top{a.k}_union_UB"] = 100 * np.mean([S[t, list(pick[ids[t]])].max() for t in range(len(ids))])

    print(f"\n=== QASC val {len(ids)} — Mixture-of-LoRA (k={a.k}, {a.combo}, w={a.weights or 'equal'}, route={a.route}) ===")
    print(f"  top-1 confidence 단일생성 : {single_acc:.1f}%")
    print(f"  ★ top-{a.k} 병합 단일생성(배포): {merged_acc:.1f}%   ← 핵심 지표")
    for k, v in refs.items():
        print(f"  ({k:16}) : {v:.1f}%")
    if "best_single_real" in refs:
        print(f"  → 병합 − best-single : {merged_acc-refs['best_single_real']:+.1f}pp "
              f"({'앙상블 이득' if merged_acc>refs['best_single_real'] else '간섭'})")
    # 출력 저장
    out = REPO / "results/qasc/seed20210211" / f"moe_merge_k{a.k}_{a.combo}_{a.weights.replace(',','-') or 'equal'}_{a.route}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "pair": [experts[pick[i][0]], experts[pick[i][1]]],
                                "merged": merged[i], "single": single[i],
                                "gold": src[i]["ground_truth"]}, ensure_ascii=False) + "\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

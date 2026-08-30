#!/usr/bin/env python3
"""라우터 입력 특징 비교용 encoder 임베딩 추출 (acc train/test).

base LLM hidden-state(extract_hidden_states.py)와 대조하기 위해, 순수 의미 임베딩
(embed_test_viz.py의 tSNE에 쓰던 것과 동일 모델 google/embeddinggemma-300m)을
router_common.py의 DatasetSpec.emb_train/emb_eval 경로에 맞춰 저장한다.

기본 동작은 SPECS의 train/eval 원본(=SFT 서브셋 7,079 + test 751)을 뽑는 것이고,
`--src`를 주면 그 jsonl 하나만 임의 경로로 뽑는다(진화 train 전체 11,097 등 spec 밖 집합용).

Usage:
  python scripts/extract_embed_acc.py
  python scripts/extract_embed_acc.py --src export/acc_v2/acc_train.jsonl \
      --out_npy results/embed_viz_test/acc_trainfull_emb.npy \
      --out_ids results/embed_viz_test/acc_trainfull_emb_ids.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

REPO = rc.REPO
EMBED_MODEL = "google/embeddinggemma-300m"


def embed(texts: list[str], device: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"임베딩 {len(texts):,}개 on {device}", flush=True)
    return np.asarray(model.encode([t[:6000] for t in texts],
                                   batch_size=256 if device == "cuda" else 16,
                                   show_progress_bar=True, normalize_embeddings=True,
                                   prompt_name="Clustering"), dtype=np.float32)


def dump(rows: list, emb_npy: Path, ids_p: Path, device: str):
    ids = [str(r["id"]) for r in rows]
    emb = embed([r["instruction"] for r in rows], device)
    emb_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_npy, emb)
    json.dump(ids, open(ids_p, "w"))
    print(f"done -> {emb_npy}  shape {emb.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="", help="지정 시 이 jsonl 하나만 뽑는다(repo 상대 경로)")
    ap.add_argument("--out_npy", default="")
    ap.add_argument("--out_ids", default="")
    a = ap.parse_args()
    sp = rc.spec("acc")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.src:
        if not (a.out_npy and a.out_ids):
            raise SystemExit("--src를 쓰면 --out_npy/--out_ids도 지정해야 한다")
        rows = [json.loads(l) for l in open(REPO / a.src, encoding="utf-8")]
        dump(rows, REPO / a.out_npy, REPO / a.out_ids, device)
        return
    for which, split in (("train", sp.train_split), ("eval", sp.eval_split)):
        rows = [json.loads(l) for l in open(REPO / sp.src[split], encoding="utf-8")]
        emb_npy, ids_p = rc.emb_paths(sp, which)
        dump(rows, emb_npy, ids_p, device)


if __name__ == "__main__":
    main()

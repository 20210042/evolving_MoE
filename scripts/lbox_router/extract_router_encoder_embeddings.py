#!/usr/bin/env python3
"""Extract normalized EmbeddingGemma features for an LBox router split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("results/embed_viz_test"))
    args = parser.parse_args()

    source = Path(f"export/lbox/lbox_{args.split}.jsonl")
    rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    ids = [str(row["id"]) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / f"lbox_{args.split}_encoder.npy"
    ids_path = args.output_dir / f"lbox_{args.split}_encoder_ids.json"
    if feature_path.exists() and ids_path.exists():
        if json.loads(ids_path.read_text(encoding="utf-8")) == ids:
            print(f"Valid cache: {feature_path}")
            return

    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for encoder feature extraction")
    model = SentenceTransformer("google/embeddinggemma-300m", device="cuda")
    options = {"normalize_embeddings": True, "show_progress_bar": True}
    if getattr(model, "prompts", None) and "Clustering" in model.prompts:
        options["prompt_name"] = "Clustering"
    embeddings = model.encode(
        [row["instruction"][:6000] for row in rows],
        batch_size=args.batch,
        **options,
    )
    np.save(feature_path, np.asarray(embeddings, dtype=np.float32))
    ids_path.write_text(json.dumps(ids) + "\n", encoding="utf-8")
    print(f"Wrote {feature_path}: {len(ids)} x {embeddings.shape[1]}")


if __name__ == "__main__":
    main()

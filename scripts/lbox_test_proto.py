#!/usr/bin/env python3
"""lbox test tSNE 프로토타입: fine-tuned dense Llama가 test에서 뭘 풀고 못 푸는지 한 장."""
import ast, json
from pathlib import Path
import numpy as np

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
OUT = REPO / "results" / "embed_viz"
SRC = REPO / "lbox_Llama-3.1-8B-Instruct_baseline_208397.jsonl"

rows = []
for l in open(SRC, encoding="utf-8"):
    r = json.loads(l)
    try:
        msgs = ast.literal_eval(r["input"])
        instr = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    except Exception:
        instr = str(r.get("input", ""))
    rows.append({"id": r["id"], "instr": instr, "pass": float(r["pass_score"]),
                 "tt": "casename" if r["id"].startswith("casename") else "statute"})
print(f"rows={len(rows)}")

emb_p = OUT / "lbox_test_emb.npy"
if emb_p.is_file() and len(np.load(emb_p)) == len(rows):
    emb = np.load(emb_p)
    print("emb cache hit")
else:
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("google/embeddinggemma-300m", device=device)
    print(f"embedding on {device}")
    emb = model.encode([x["instr"][:6000] for x in rows],
                       batch_size=256 if device == "cuda" else 32,
                       show_progress_bar=True, normalize_embeddings=True,
                       prompt_name="Clustering")
    emb = np.asarray(emb, dtype=np.float32)
    np.save(emb_p, emb)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
x50 = PCA(50, random_state=42).fit_transform(emb)
xy = TSNE(2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(x50)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
solved = np.array([x["pass"] > 0 for x in rows])
acc = 100 * solved.mean()
fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.scatter(xy[solved, 0], xy[solved, 1], s=6, c="#cfcfcf", alpha=0.7, linewidths=0,
           label=f"fine-tuned SOLVED ({int(solved.sum()):,})")
ax.scatter(xy[~solved, 0], xy[~solved, 1], s=6, c="#111111", alpha=0.6, linewidths=0,
           label=f"fine-tuned FAILED ({int((~solved).sum()):,})")
ax.set_title(f"LBOX test ({len(rows):,}) — fine-tuned dense Llama3 pass/fail  (acc {acc:.1f}%)",
             fontsize=12, color="#0b0b0b")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
          markerscale=2.5, fontsize=9)
fig.tight_layout()
out = OUT / "lbox_test_proto.png"
fig.savefig(out, dpi=160, facecolor="#fcfcfb", bbox_inches="tight")
print(f"saved -> {out}  (acc {acc:.1f}%)")

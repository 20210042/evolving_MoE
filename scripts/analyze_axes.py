#!/usr/bin/env python3
"""논문 셀링포인트 정량화 — 두 분석.

분석 1 (centroid / semantic-routing 테스트, 재훈 2-1 직접 검증):
  expert 프롬프트를 임베딩해 centroid로 삼고, 각 문제를 가장 가까운 프롬프트에 배정
  (= 프롬프트가 '약속하는' 라우팅). 이 배정이
    (a) human prior와 얼마나 닮았나 (ARI, 높을 것 — 둘 다 의미 기반)
    (b) 실제로 그 expert가 그 문제를 푸나 (solve-rate, random 배정과 비교)
  → semantic ≈ random 이면 "프롬프트 의미는 실제 solve와 무관" = 2-1 확증.

분석 2 (solvability 클러스터링, 재훈 2-1 TODO):
  문제별 solve-signature(누가 풀었나 이진벡터)로 문제를 클러스터 →
    (a) 실재하는 구조인가
    (b) 임베딩 기하/human prior와 직교하나 (ARI ≈ 0 기대)
  → solvability 축이 시맨틱 축과 다른 실제 구조임을 제시.

LUCA(generic 프롬프트)는 specialist 라우팅 후보에서 제외.
Usage: python scripts/analyze_axes.py --dataset qasc|lbox
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")

CFG = {
    "qasc": dict(
        labels="export/qasc_binning_seed20210211/binning_labels.jsonl",
        mapping="export/qasc_binning_seed20210211/agent_mapping.json",
        src="export/qasc/qasc_train.jsonl",
        emb="results/embed_viz/qasc_emb.npy",
        prior_tags="results/embed_viz/qasc_llm_tags.json",  # id -> subject
        prior_fn=None,
    ),
    "lbox": dict(
        labels="export/lbox_binning_seed20210311/binning_labels.jsonl",
        mapping="export/lbox_binning_seed20210311/agent_mapping.json",
        src="export/lbox/lbox_train.jsonl",
        emb="results/embed_viz/lbox_emb.npy",
        prior_tags=None,
        prior_fn=lambda r: f"{r.get('task_type')}·{r.get('casetype')}",
    ),
}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CFG), default="qasc")
    a = ap.parse_args()
    c = CFG[a.dataset]

    labels = load_jsonl(REPO / c["labels"])
    id2lab = {str(r["id"]): r for r in labels}
    mapping = json.load(open(REPO / c["mapping"], encoding="utf-8"))
    experts = sorted({e for r in labels for e in r["per_expert"]})
    specialists = [e for e in experts if e != "luca"]  # LUCA 제외

    src = load_jsonl(REPO / c["src"])
    src_ids = [str(r["id"]) for r in src]
    src_by_id = {str(r["id"]): r for r in src}
    emb = np.load(REPO / c["emb"])
    assert len(emb) == len(src_ids), f"emb {len(emb)} != src {len(src_ids)}"
    # 라벨 있는 문제만
    keep = [i for i, pid in enumerate(src_ids) if pid in id2lab]
    ids = [src_ids[i] for i in keep]
    P = emb[keep]
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    n = len(ids)

    # human prior 라벨
    if c["prior_tags"]:
        tags = json.load(open(REPO / c["prior_tags"], encoding="utf-8"))
        prior = np.array([tags.get(i, "?") for i in ids])
    else:
        prior = np.array([c["prior_fn"](src_by_id[i]) for i in ids])

    # solve-signature (specialist만): 문제 × specialist 이진
    S = np.array([[int(id2lab[i]["per_expert"].get(e, 0)) for e in specialists] for i in ids])
    n_solved = S.sum(1)

    print(f"\n{'='*66}\n{a.dataset.upper()} — {n:,} problems · {len(specialists)} specialists (LUCA 제외)\n{'='*66}")

    # ---- expert 프롬프트 임베딩 (centroid) ----
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("google/embeddinggemma-300m", device="cpu")
    proto_txt = [mapping[e].get("system_prompt", "") or mapping[e].get("name", e) for e in specialists]
    kw = {"prompt_name": "Clustering"} if getattr(model, "prompts", None) and "Clustering" in model.prompts else {}
    C = np.asarray(model.encode(proto_txt, normalize_embeddings=True, **kw), dtype=np.float32)

    # ---- 분석 1: semantic routing ----
    sim = P @ C.T                    # (n, n_spec) 코사인
    route = sim.argmax(1)            # 각 문제가 배정된 specialist index
    route_e = [specialists[j] for j in route]

    per_expert_rate = {e: S[:, k].mean() for k, e in enumerate(specialists)}
    semantic_solve = np.mean([S[p, route[p]] for p in range(n)])
    random_solve = np.mean(list(per_expert_rate.values()))   # 무작위 배정 기대 solve-rate
    oracle_solve = (n_solved > 0).mean()
    best_solve = max(per_expert_rate.values())

    print("\n[분석 1] semantic-routing (프롬프트 centroid 최근접 배정)")
    print(f"  semantic route solve-rate : {100*semantic_solve:5.1f}%")
    print(f"  random  route solve-rate  : {100*random_solve:5.1f}%   (specialist 평균 pass)")
    print(f"  best single expert        : {100*best_solve:5.1f}%")
    print(f"  oracle (union)            : {100*oracle_solve:5.1f}%")
    lift = 100 * (semantic_solve - random_solve)
    print(f"  → semantic − random lift  : {lift:+.1f}pp  "
          f"({'무관(2-1 확증)' if abs(lift) < 2 else '의미 있음'})")
    # 배정 파티션이 human prior와 닮았나 vs solve와 닮았나
    best_solver = np.array([specialists[int(np.argmax([S[p, k] * per_expert_rate[specialists[k]]**-1
                            if S[p, k] else -1 for k in range(len(specialists))]))]
                            if n_solved[p] > 0 else "(none)" for p in range(n)])
    print(f"  ARI(semantic-route, human prior) : {ARI(route_e, prior):.3f}   ← 의미끼리")
    print(f"  ARI(semantic-route, solve-best)  : {ARI(route_e, best_solver):.3f}   ← solve와")

    # ---- 분석 2: solvability 클러스터링 ----
    print("\n[분석 2] solvability 클러스터링 (solve-signature 공간)")
    # 임베딩 기하 파티션(비교용): PCA→KMeans
    from sklearn.decomposition import PCA
    Pk = PCA(min(50, P.shape[1]), random_state=42).fit_transform(P)
    kside = min(len(specialists), 10)
    emb_cluster = KMeans(kside, n_init=5, random_state=42).fit_predict(Pk)

    for scope, mask in [("전체", np.ones(n, bool)),
                        ("contested(0<n_solved<N)", (n_solved > 0) & (n_solved < len(specialists)))]:
        idx = np.where(mask)[0]
        if len(idx) < 50:
            continue
        sig = S[idx]
        solve_cluster = KMeans(kside, n_init=5, random_state=42).fit_predict(sig.astype(float))
        # 응집도: 같은 solve-cluster 내 signature 동일성(평균 Hamming 유사도)
        coh = np.mean([1 - (np.abs(sig[solve_cluster == g] - sig[solve_cluster == g].mean(0)) ).mean()
                       for g in np.unique(solve_cluster)])
        ari_geo = ARI(solve_cluster, emb_cluster[idx])
        ari_pri = ARI(solve_cluster, prior[idx])
        print(f"  [{scope}] {len(idx):,}개, {kside} clusters | 응집도 {coh:.2f} | "
              f"ARI vs 임베딩기하 {ari_geo:.3f} | ARI vs human prior {ari_pri:.3f}")

    print(f"\n  참고: ARI(human prior, 임베딩기하) = {ARI(prior, emb_cluster):.3f}  ← 시맨틱 두 축은 서로 정렬")
    print()


if __name__ == "__main__":
    main()

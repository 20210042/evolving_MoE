#!/usr/bin/env python3
"""라우터 실험 공통: 데이터셋 레지스트리 + 경로/정렬 헬퍼.

라우터 스크립트들이 각자 절대경로와 QASC 파일명을 하드코딩하고 있어서 다른
도메인에 못 쓰던 걸 여기로 모았다. 새 도메인은 SPECS에 한 줄 추가하면 된다.

특징 파일 명명 규칙 (results/embed_viz_test/):
    <ds>_<split>_hs_mean.npy / _hs_last.npy / _hs_ids.json   extract_hidden_states.py
    <ds>_<split>_ansprob.npy / _ansprob_ids.json              extract_answer_logits.py
    <ds>_<split>_expert_conf.npz                              extract_expert_confidence.py
encoder 임베딩만 예전 레이아웃이 남아 있어 spec에 경로를 직접 적는다.

⚠️ ansprob / expert_conf 는 **MCQA 전용**이다 — 다음 토큰 위치에서 정답 레터
(A~H) 분포를 읽는 방식이라 고정된 답 어휘가 있어야 성립한다. LBOX처럼 답을
생성해 문자열로 채점하는 오픈 QA에서는 그대로 못 쓰고 정의를 새로 세워야 한다.
`answer_letters is None` 인 spec에서 이 경로를 타면 require_mcqa()가 막는다.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FEAT_DIR = REPO / "results" / "embed_viz_test"

sys.path.insert(0, str(REPO / "src"))
from prompts import baseline_prompts as bp  # noqa: E402
from prompts.coding import sni_system_block, sni_user_block  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_split: str
    eval_split: str
    src: dict                  # split -> 원본 jsonl (repo 상대)
    labels: str                # train binning 라벨 (학습 타깃)
    eval_binned: str           # eval split binned 라벨
    ckpt: str                  # per-expert 어댑터 디렉터리
    base_model: str
    gen: tuple                 # (system prompt, user template)
    answer_letters: tuple | None   # MCQA 답 어휘. None이면 오픈 QA
    emb_train: tuple           # (npy, ids) — encoder 임베딩만 레거시 경로
    emb_eval: tuple
    # gen 템플릿 하나로 조립되지 않는 도메인용. row -> (system, user).
    # None이면 기존대로 gen=(system, user_template.format(instruction=...)).
    build_prompt: object = None


SPECS = {
    "qasc": DatasetSpec(
        name="qasc",
        train_split="train",
        eval_split="validation",
        src={"train": "export/qasc/qasc_train.jsonl",
             "validation": "export/qasc/qasc_validation.jsonl"},
        labels="export/qasc_binning_seed20210211/binning_labels.jsonl",
        eval_binned="results/qasc/seed20210211/inference_validation_binning_final.binned.jsonl",
        ckpt="checkpoints/expert_sft/qasc_seed20210211_cap10",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        gen=(bp.QASC_GEN_SYSTEM, bp.QASC_GEN_USER),
        answer_letters=tuple("ABCDEFGH"),
        emb_train=("results/embed_viz/qasc_emb.npy", "results/embed_viz/qasc_ids.json"),
        emb_eval=("results/embed_viz_test/qasc_val_emb.npy",
                  "export/qasc/qasc_validation.jsonl"),
    ),
    "acc": DatasetSpec(
        name="acc",
        train_split="train",
        eval_split="test",
        src={"train": "export/acc_v2/sft/acc_train.jsonl",
             "test": "export/acc_v2/acc_test.jsonl"},
        labels="export/acc_binning_seed20210111_v2/binning_labels.jsonl",
        eval_binned="results/acc/seed20210111_v2/ablation/inference_test751_evolved_fewshot.binned.jsonl",
        ckpt="checkpoints/expert_sft/acc_seed20210111_v2_cap9_fewshot",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        gen=(bp.CODING_GEN_SYSTEM, bp.CODING_GEN_USER),
        answer_letters=None,     # 코드실행 채점 — 답을 생성해 테스트케이스로 채점
        emb_train=("results/embed_viz_test/acc_train_emb.npy", "results/embed_viz_test/acc_train_emb_ids.json"),
        emb_eval=("results/embed_viz_test/acc_test_emb.npy", "results/embed_viz_test/acc_test_emb_ids.json"),
    ),
    "lbox": DatasetSpec(
        name="lbox",
        train_split="train",
        eval_split="valid",
        src={"train": "export/lbox/lbox_train.jsonl",
             "valid": "export/lbox/lbox_valid.jsonl",
             "test": "export/lbox/lbox_test.jsonl"},
        labels="export/lbox_binning_seed20210311/binning_labels.jsonl",
        eval_binned="results/lbox/seed20210311/inference_valid_binning_final.binned.jsonl",
        ckpt="checkpoints/expert_sft/lbox_seed20210311_cap10",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        gen=(bp.LBOX_GEN_SYSTEM, bp.LBOX_GEN_USER),
        answer_letters=None,     # 오픈 QA — 답을 생성해 문자열 EM으로 채점
        emb_train=("results/embed_viz/lbox_emb.npy", "results/embed_viz/lbox_ids.json"),
        emb_eval=("results/embed_viz_test/lbox_valid_emb.npy",
                  "export/lbox/lbox_valid.jsonl"),
    ),
    "sni": DatasetSpec(
        name="sni",
        train_split="train",
        eval_split="test",
        src={"train": "export/sni_v4/sni_train.jsonl",
             "valid": "export/sni_v4/sni_valid.jsonl",
             "test": "export/sni_v4/sni_test.jsonl"},
        labels="export/sni_binning_seed20212003/binning_labels.jsonl",
        eval_binned="export/sni_binning_seed20212003/test_binned.jsonl",
        ckpt="",                 # SNI는 per-expert 어댑터가 없다 — 로스터가 프롬프트 페르소나다
        base_model="google/gemma-4-26B-A4B-it",
        gen=(bp.SNI_GEN_SYSTEM, bp.SNI_GEN_USER),
        answer_letters=None,     # 오픈 생성 — 공식 EM/ROUGE로 채점
        emb_train=("results/embed_viz_test/sni_train_emb.npy",
                   "results/embed_viz_test/sni_train_emb_ids.json"),
        emb_eval=("results/embed_viz_test/sni_test_emb.npy",
                  "results/embed_viz_test/sni_test_emb_ids.json"),
        # 실제 생성과 같은 조립. 페르소나 자리는 중립(SNI_GEN_SYSTEM) — 전문가 정보는
        # 프로파일 벡터로 따로 들어간다(scripts/extract_expert_profiles.py).
        build_prompt=lambda r: (
            sni_system_block(bp.SNI_GEN_SYSTEM, r.get("definition")),
            sni_user_block(r.get("answer_line"), r["instruction"],
                           positive_examples=r.get("positive_examples"), num_pos=2),
        ),
    ),
}


def spec(name: str) -> DatasetSpec:
    if name not in SPECS:
        raise SystemExit(f"모르는 데이터셋 '{name}' — 등록된 것: {sorted(SPECS)}")
    return SPECS[name]


def add_dataset_arg(ap, default="qasc"):
    ap.add_argument("--dataset", default=default, choices=sorted(SPECS))
    return ap


def require_mcqa(sp: DatasetSpec, what: str):
    """MCQA 전용 특징을 오픈 QA에서 쓰려 할 때 조용히 틀린 값을 내지 않도록 막는다."""
    if sp.answer_letters is None:
        raise NotImplementedError(
            f"{what}은(는) 현재 MCQA 전용이다 — 다음 토큰의 정답 레터 분포를 쓰기 때문에 "
            f"고정된 답 어휘가 필요하다. '{sp.name}'은 답을 생성해 문자열로 채점하는 "
            f"오픈 QA라 이 정의가 성립하지 않으므로, 쓰려면 확신도 정의부터 새로 세워야 한다."
        )


# ---- 로더 ----
def load_binning(path) -> dict:
    return {str(json.loads(l)["id"]): json.loads(l)
            for l in open(path, encoding="utf-8") if l.strip()}


def load_ids(path) -> list:
    """jsonl이면 id 필드, json이면 리스트 그대로."""
    p = str(path)
    if p.endswith(".jsonl"):
        return [str(json.loads(l)["id"]) for l in open(p, encoding="utf-8") if l.strip()]
    return [str(x) for x in json.load(open(p, encoding="utf-8"))]


def experts(sp: DatasetSpec) -> list:
    with open(REPO / sp.labels, encoding="utf-8") as f:
        return sorted(json.loads(f.readline())["per_expert"])


def labels(sp: DatasetSpec):
    """(train binning, eval binned) 딕셔너리 쌍."""
    return load_binning(REPO / sp.labels), load_binning(REPO / sp.eval_binned)


# ---- 특징 경로 ----
def feat_path(sp: DatasetSpec, split: str, kind: str) -> Path:
    """kind: hs_mean | hs_last | hs_ids | ansprob | ansprob_ids | conf"""
    if kind in ("ansprob", "ansprob_ids", "conf"):
        require_mcqa(sp, f"'{kind}' 특징")
    tag = f"{sp.name}_{split}"
    return FEAT_DIR / {
        "hs_mean": f"{tag}_hs_mean.npy",
        "hs_last": f"{tag}_hs_last.npy",
        "hs_ids": f"{tag}_hs_ids.json",
        "ansprob": f"{tag}_ansprob.npy",
        "ansprob_ids": f"{tag}_ansprob_ids.json",
        "conf": f"{tag}_expert_conf.npz",
    }[kind]


def emb_paths(sp: DatasetSpec, which: str):
    """which: train | eval → (npy 절대경로, ids 절대경로)."""
    npy, ids = sp.emb_train if which == "train" else sp.emb_eval
    return REPO / npy, REPO / ids


# ---- 정렬 ----
def align(feat, ids_path, binning, ex, order=None):
    """특징 행렬과 solve 행렬을 같은 id 순서로 맞춘다.

    order를 주면 그 순서를 강제한다(특징 여러 개를 concat할 때 필요).
    """
    F = np.load(feat)
    ids = load_ids(ids_path)
    idx = {p: i for i, p in enumerate(ids)}
    use = list(order) if order is not None else [p for p in ids if p in binning]
    X = np.array([F[idx[p]] for p in use], np.float32)
    S = np.array([[binning[p]["per_expert"].get(e, 0) for e in ex] for p in use], np.float32)
    return use, X, S


def align_conf(sp: DatasetSpec, split: str, ex, order):
    """expert confidence npz를 (문제 order × ex) 로 정렬."""
    d = np.load(feat_path(sp, split, "conf"), allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    exps = [str(x) for x in d["experts"]]
    ii = {p: i for i, p in enumerate(ids)}
    ej = {e: j for j, e in enumerate(exps)}
    return np.array([[d["conf"][ii[p], ej[e]] for e in ex] for p in order], np.float32)


# ---- 기준선 ----
def baselines(S):
    """(best-single expert %, oracle union %). 하드코딩하지 말고 이걸 쓸 것 —
    라벨이 바뀌면 기준선도 같이 바뀐다."""
    return 100 * S.mean(0).max(), 100 * (S.sum(1) > 0).mean()


def anchor_expert(S, ex) -> int:
    """anchor/residual 실험용 best-single expert 인덱스 (train solve율 기준)."""
    return int(S.mean(0).argmax())


def zscore(X):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    return mu, sd

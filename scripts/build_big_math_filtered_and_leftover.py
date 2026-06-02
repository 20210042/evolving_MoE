"""
BIG-MATH_filtered 소규모 균형 버전 생성 스크립트

카테고리별 정확히 10,000 / 2,000 / 2,000개로 서브샘플링:
  - algebra
  - geometry
  - statistics_and_probability
  - calculus_and_precalculus
  - number_theory_and_discrete_math

총 50,000 (train) / 10,000 (val) / 10,000 (test) = 70,000개
"""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from collections import Counter
from pathlib import Path

# ── 설정 ─────────────────────────────────────────────────────────────────────
SEED = 42

N_TRAIN_PER_CAT = 10_000
N_VAL_PER_CAT   =  2_000
N_TEST_PER_CAT  =  2_000
N_PER_CAT       = N_TRAIN_PER_CAT + N_VAL_PER_CAT + N_TEST_PER_CAT  # 14,000

CATEGORIES = [
    "algebra",
    "geometry",
    "number_theory_and_discrete_math",
    "calculus_and_precalculus",
    "statistics_and_probability",
]

_ROOT    = Path(__file__).resolve().parent.parent
OUT_PATH = str(_ROOT / "big_math_filtered_small")

# ── 카테고리 분류 함수 (build_big_math_datasets.py와 동일) ──────────────────
MERGE_MAP = {
    "algebra":                    "algebra",
    "geometry":                   "geometry",
    "statistics_and_probability": "statistics_and_probability",
    "calculus":                   "calculus_and_precalculus",
    "precalculus":                "calculus_and_precalculus",
    "number_theory":              "number_theory_and_discrete_math",
    "discrete_mathematics":       "number_theory_and_discrete_math",
}

STATS_PROB = {"Statistics", "Probability"}
LEVEL2_MAP = {
    "Algebra":              "algebra",
    "Geometry":             "geometry",
    "Calculus":             "calculus",
    "Number Theory":        "number_theory",
    "Discrete Mathematics": "discrete_mathematics",
    "Precalculus":          "precalculus",
}


def filter_and_merge_to_7_categories(domain_list: list) -> list:
    cats = []
    for d in domain_list:
        parts = [p.strip() for p in d.split("->")]
        if len(parts) < 2 or parts[0] != "Mathematics":
            continue
        l2 = parts[1]
        if l2 == "Applied Mathematics":
            l3 = parts[2] if len(parts) >= 3 else None
            if l3 in STATS_PROB:
                c = "statistics_and_probability"
                if c not in cats:
                    cats.append(c)
        elif l2 in LEVEL2_MAP:
            c = LEVEL2_MAP[l2]
            if c not in cats:
                cats.append(c)
    return cats


def merge_to_5(original_cats: list) -> list:
    merged = []
    for c in original_cats:
        m = MERGE_MAP.get(c)
        if m and m not in merged:
            merged.append(m)
    return merged


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("Loading SynthLabsAI/Big-Math-RL-Verified ...")
ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
df = ds.to_pandas()
print(f"  원본 샘플 수: {len(df):,}")

# ── 카테고리 컬럼 생성 ────────────────────────────────────────────────────────
df["category"] = df["domain"].apply(filter_and_merge_to_7_categories).apply(merge_to_5)

# single-category 샘플만
mask_single = df["category"].apply(len) == 1
df_filtered = df[mask_single].copy()
df_leftover = df[~mask_single].copy()
print(f"  single-label 샘플 수: {len(df_filtered):,}")
print(f"  leftover 샘플 수:     {len(df_leftover):,}")

COLS = ["problem", "answer", "source", "domain", "category", "llama8b_solve_rate"]
df_filtered = df_filtered[COLS].rename(columns={"domain": "original_domain"})
df_leftover = df_leftover[COLS].rename(columns={"domain": "original_domain"})

# ── 카테고리별 서브샘플링 후 10k / 2k / 2k 분할 ──────────────────────────────
train_parts, val_parts, test_parts = [], [], []

print(f"\n  카테고리별 서브샘플링 ({N_PER_CAT:,}개 → {N_TRAIN_PER_CAT:,}/{N_VAL_PER_CAT:,}/{N_TEST_PER_CAT:,}):")
for cat in CATEGORIES:
    df_cat = df_filtered[df_filtered["category"].apply(lambda x: x[0]) == cat]
    df_cat = df_cat.sample(frac=1, random_state=SEED).reset_index(drop=True)

    available = len(df_cat)
    if available < N_PER_CAT:
        print(f"  WARNING: {cat} 샘플 부족 ({available:,} < {N_PER_CAT:,}) — 전부 사용")
    else:
        df_cat = df_cat.iloc[:N_PER_CAT]

    n_train = min(N_TRAIN_PER_CAT, len(df_cat))
    n_val   = min(N_VAL_PER_CAT,   len(df_cat) - n_train)
    n_test  = len(df_cat) - n_train - n_val

    train_parts.append(df_cat.iloc[:n_train])
    val_parts.append(df_cat.iloc[n_train:n_train + n_val])
    test_parts.append(df_cat.iloc[n_train + n_val:])

    print(f"    {cat}: train={n_train:,} / val={n_val:,} / test={n_test:,}")

# ── 각 split을 합치고 shuffle ─────────────────────────────────────────────────
df_train = pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_val   = pd.concat(val_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_test  = pd.concat(test_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\n  split → train: {len(df_train):,} / val: {len(df_val):,} / test: {len(df_test):,}")

# 분포 확인
for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
    cat_dist = Counter(part["category"].apply(lambda x: x[0]))
    print(f"  [{name}] " + ", ".join(f"{k}={v:,}" for k, v in sorted(cat_dist.items(), key=lambda x: -x[1])))

# ── HuggingFace Dataset 생성 ──────────────────────────────────────────────────
ds_filtered = DatasetDict({
    "train":      Dataset.from_pandas(df_train,    preserve_index=False),
    "validation": Dataset.from_pandas(df_val,      preserve_index=False),
    "test":       Dataset.from_pandas(df_test,     preserve_index=False),
})
ds_leftover = Dataset.from_pandas(df_leftover, preserve_index=False)

# ── 로컬 저장 ─────────────────────────────────────────────────────────────────
OUT_LEFTOVER = str(_ROOT / "big_math_leftover")
print(f"\nSaving to disk ...")
ds_filtered.save_to_disk(OUT_PATH)
ds_leftover.save_to_disk(OUT_LEFTOVER)

# ── HuggingFace Hub 업로드 ────────────────────────────────────────────────────
print("\nUploading to HuggingFace Hub ...")
ds_filtered.push_to_hub("Jongbin-kr/BIG-MATH_filtered", private=False)
print("  BIG-MATH_filtered 완료.")
ds_leftover.push_to_hub("Jongbin-kr/BIG-MATH_leftover", private=False)
print("  BIG-MATH_leftover 완료.")

print("\nDone.")

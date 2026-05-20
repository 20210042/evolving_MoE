"""
Big-Math-RL-Verified 데이터셋 전처리 스크립트

5개 카테고리로 재구성:
  - algebra
  - geometry
  - statistics_and_probability
  - calculus_and_precalculus     (calculus + precalculus 합산)
  - number_theory_and_discrete_math  (number_theory + discrete_mathematics 합산)

컬럼 구조:
  - original_domain : 원본 domain 칼럼 그대로
  - category        : 5-class 카테고리 리스트 (신규)

big_math_filtered : 단일 카테고리 샘플만, train/val/test 10:1:1 split
big_math_leftover : 나머지 전체 (다중 카테고리 포함), split 없음
"""

import pyarrow as pa
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from collections import Counter
from pathlib import Path

# ── 설정 ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
OUT_FILTERED  = str(_ROOT / "big_math_filtered")
OUT_LEFTOVER  = str(_ROOT / "big_math_leftover")

SPLIT_RATIO = (10, 1, 1)  # train : val : test


# 7개 → 5개 카테고리 매핑
MERGE_MAP = {
    "algebra":                  "algebra",
    "geometry":                 "geometry",
    "statistics_and_probability": "statistics_and_probability",
    "calculus":                 "calculus_and_precalculus",
    "precalculus":              "calculus_and_precalculus",
    "number_theory":            "number_theory_and_discrete_math",
    "discrete_mathematics":     "number_theory_and_discrete_math",
}


# ── 원본 분류 함수 (기존과 동일) ─────────────────────────────────────────────
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
    """domain 리스트 → 7-class 카테고리 리스트 (중복 제거)"""
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
    """7-class → 5-class 매핑 (중복 제거)"""
    merged = []
    for c in original_cats:
        m = MERGE_MAP.get(c)
        if m and m not in merged:
            merged.append(m)
    return merged

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("Loading arrow file...")
ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified",split="train")
df = ds.to_pandas()
print(f"  원본 샘플 수: {len(df):,}")

# ── 카테고리 컬럼 생성 ────────────────────────────────────────────────────────
df["category"] = df["domain"].apply(filter_and_merge_to_7_categories).apply(merge_to_5)

# ── filtered: 5-class 중 정확히 1개인 샘플 ────────────────────────────────────
mask_single = df["category"].apply(len) == 1
df_filtered  = df[mask_single].copy()
df_leftover  = df[~mask_single].copy()

print(f"\n  single-label (filtered): {len(df_filtered):,}")
print(f"  나머지 (leftover):        {len(df_leftover):,}")

# filtered 카테고리 분포
cat_counts = Counter(df_filtered["category"].apply(lambda x: x[0]))
print("\n  [filtered] 카테고리별 샘플 수:")
for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"    {k}: {v:,}")

# ── 컬럼 정리 ─────────────────────────────────────────────────────────────────
COLS = ["problem", "answer", "source", "domain", "category", "llama8b_solve_rate"]
df_filtered = df_filtered[COLS].rename(columns={"domain": "original_domain"})
df_leftover = df_leftover[COLS].rename(columns={"domain": "original_domain"})

# ── shuffle 후 train / val / test 분할 ───────────────────────────────────────
df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

total  = len(df_filtered)
n_val  = total // sum(SPLIT_RATIO)   # 1/12
n_test = total // sum(SPLIT_RATIO)   # 1/12
n_train = total - n_val - n_test

df_train = df_filtered.iloc[:n_train]
df_val   = df_filtered.iloc[n_train:n_train + n_val]
df_test  = df_filtered.iloc[n_train + n_val:]

print(f"\n  split → train: {len(df_train):,} / val: {len(df_val):,} / test: {len(df_test):,}")

# 분포 확인
for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
    cat_dist = Counter(part["category"].apply(lambda x: x[0]))
    src_dist  = Counter(part["source"])
    print(f"\n  [{name}] 카테고리: " + ", ".join(f"{k}={v:,}" for k, v in sorted(cat_dist.items(), key=lambda x: -x[1])))
    print(f"  [{name}] 출처: "     + ", ".join(f"{k}={v:,}" for k, v in sorted(src_dist.items(),  key=lambda x: -x[1])))

# ── HuggingFace Dataset 생성 ──────────────────────────────────────────────────
ds_filtered = DatasetDict({
    "train":      Dataset.from_pandas(df_train, preserve_index=False),
    "validation": Dataset.from_pandas(df_val,   preserve_index=False),
    "test":       Dataset.from_pandas(df_test,  preserve_index=False),
})
ds_leftover = Dataset.from_pandas(df_leftover, preserve_index=False)

# ── 로컬 저장 ─────────────────────────────────────────────────────────────────
print("\nSaving to disk ...")
ds_filtered.save_to_disk(OUT_FILTERED)
ds_leftover.save_to_disk(OUT_LEFTOVER)
print(f"  {OUT_FILTERED}")
print(f"  {OUT_LEFTOVER}")

# ── HuggingFace Hub 업로드 ────────────────────────────────────────────────────
print("\nUploading to HuggingFace Hub ...")
ds_filtered.push_to_hub("Jongbin-kr/BIG-MATH_filtered",  private=False)
ds_leftover.push_to_hub("Jongbin-kr/BIG-MATH_leftover", private=False)
print("  업로드 완료.")

print("\nDone.")

"""
NuminaMath-CoT_filtered 데이터셋 생성 스크립트

1. AI-MO/NuminaMath-1.5의 problem_type을 키로 AI-MO/NuminaMath-CoT에 카테고리 라벨 부여
   (조인 키: (problem, source) → fallback: problem)
2. 5개 카테고리만 사용: Algebra, Geometry, Number Theory, Combinatorics, Calculus
3. 가장 적은 Calculus 기준으로 카테고리별 동일 수 샘플링
4. train:val:test = 10:1:1
5. problem / solution prefix 클리닝
6. Jongbin-kr/NuminaMath-CoT_filtered 로 push
"""

import pandas as pd
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset

SEED     = 42
HUB_REPO = "Jongbin-kr/NuminaMath-CoT_filtered"

CATEGORIES = ["Algebra", "Geometry", "Number Theory", "Combinatorics", "Calculus"]
CAT_SLUG = {
    "Algebra":       "algebra",
    "Geometry":      "geometry",
    "Number Theory": "number_theory",
    "Combinatorics": "combinatorics",
    "Calculus":      "calculus",
}

OUT_COLS = ["id", "problem", "solution", "category", "source", "messages"]

SYNTHETIC_SOURCES = {"synthetic_amc", "synthetic_math", "synthetic_aops", "synthetic"}



# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("Loading AI-MO/NuminaMath-1.5 ...")
ds15 = load_dataset("AI-MO/NuminaMath-1.5", split="train")
print(f"  rows: {len(ds15):,}")

print("Loading AI-MO/NuminaMath-CoT ...")
cot = load_dataset("AI-MO/NuminaMath-CoT", split="train")
print(f"  rows: {len(cot):,}")

# ── 카테고리 룩업 테이블 ──────────────────────────────────────────────────────
print("Building lookup table ...")
lookup_both: dict[tuple[str, str], str] = {}
lookup_prob: dict[str, str] = {}
for r in ds15:
    lookup_both[(r["problem"], r["source"])] = r["problem_type"]
    lookup_prob[r["problem"]]                = r["problem_type"]

# ── CoT에 카테고리 라벨 부여 ─────────────────────────────────────────────────
print("Labeling NuminaMath-CoT ...")
records = []
match_both = match_prob = no_match = 0

for r in cot:
    pt = lookup_both.get((r["problem"], r["source"]))
    if pt:
        match_both += 1
    else:
        pt = lookup_prob.get(r["problem"])
        if pt:
            match_prob += 1
        else:
            no_match += 1

    if pt in CATEGORIES and r["source"] not in SYNTHETIC_SOURCES:
        records.append({
            "problem":  r["problem"],
            "solution": r["solution"],
            "category": pt,
            "source":   r["source"],
            "messages": r["messages"],
        })

total = len(cot)
print(f"  (problem, source) match: {match_both:,} ({match_both/total*100:.1f}%)")
print(f"  problem-only match:      {match_prob:,} ({match_prob/total*100:.1f}%)")
print(f"  no match:                {no_match:,} ({no_match/total*100:.1f}%)")

df = pd.DataFrame(records)
print(f"\n  5-category rows: {len(df):,}")
print("  Category counts:")
for cat, cnt in df["category"].value_counts().items():
    print(f"    {cat}: {cnt:,}")

# ── 가장 적은 카테고리 기준 샘플 수 결정 & 10:1:1 분할 ───────────────────────
cat_counts = {cat: (df["category"] == cat).sum() for cat in CATEGORIES}
min_cat    = min(cat_counts, key=cat_counts.get)
N_TOTAL    = cat_counts[min_cat]
N_VAL      = N_TOTAL // 12
N_TEST     = N_TOTAL // 12
N_TRAIN    = N_TOTAL - N_VAL - N_TEST

print(f"\n카테고리별 가용 수:")
for cat, cnt in cat_counts.items():
    print(f"  {cat}: {cnt:,}")
print(f"\n최소 카테고리: {min_cat} ({N_TOTAL:,})  →  train={N_TRAIN:,} / val={N_VAL:,} / test={N_TEST:,}")

train_parts, val_parts, test_parts = [], [], []

for cat in CATEGORIES:
    df_cat = df[df["category"] == cat].sample(frac=1, random_state=SEED).reset_index(drop=True)
    available = len(df_cat)
    if available < N_TOTAL:
        print(f"  WARNING: {cat} has only {available:,} (need {N_TOTAL:,})")

    df_cat = df_cat.iloc[:N_TOTAL].copy()

    slug = CAT_SLUG[cat]
    df_cat["id"]       = [f"numina_cot_{slug}_{i+1}" for i in range(len(df_cat))]
    df_cat["category"] = cat

    df_train = df_cat.iloc[:N_TRAIN][OUT_COLS]
    df_val   = df_cat.iloc[N_TRAIN:N_TRAIN + N_VAL][OUT_COLS]
    df_test  = df_cat.iloc[N_TRAIN + N_VAL:][OUT_COLS]

    print(f"[{cat}] available={available:,}  →  train={len(df_train):,} / val={len(df_val):,} / test={len(df_test):,}")
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        src_dist = Counter(part["source"])
        print(f"    [{name}] " + ", ".join(f"{s}={c}" for s, c in src_dist.most_common(3)))

    train_parts.append(df_train)
    val_parts.append(df_val)
    test_parts.append(df_test)

# ── 합치기 & 최종 shuffle ─────────────────────────────────────────────────────
df_train_all = pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_val_all   = pd.concat(val_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_test_all  = pd.concat(test_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\nFinal: train={len(df_train_all):,} / val={len(df_val_all):,} / test={len(df_test_all):,}")
for name, part in [("train", df_train_all), ("val", df_val_all), ("test", df_test_all)]:
    dist = Counter(part["category"])
    print(f"  [{name}] " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))

# ── HuggingFace 업로드 ────────────────────────────────────────────────────────
ds_out = DatasetDict({
    "train":      Dataset.from_pandas(df_train_all, preserve_index=False),
    "validation": Dataset.from_pandas(df_val_all,   preserve_index=False),
    "test":       Dataset.from_pandas(df_test_all,  preserve_index=False),
})

print(f"\nColumns: {ds_out['train'].column_names}")
print(f"Uploading to {HUB_REPO} ...")
ds_out.push_to_hub(HUB_REPO, private=False)
print("Done.")

"""Original-domain label distribution counter.

Streams the large execution-ready CSV, counts empty original_domain rows and unique raw labels, then
writes the label-frequency CSV used by later normalization work."""

import pandas as pd
import ast
from pathlib import Path
from collections import Counter

# 입력 파일 경로
input_path = Path("/home/minjikim/minji_link/code/benchmark/data/processed/04_execution_ready.csv")

# 출력 파일 경로
output_path = Path("/home/minjikim/minji_link/code/benchmark/data/labelling/original_domain_unique_labels.csv")

empty_count = 0
label_counter = Counter()
chunks = pd.read_csv(
    input_path,
    usecols=["original_domain"],
    dtype={"original_domain": "string"},
    chunksize=200_000,
)

for chunk in chunks:
    for value in chunk["original_domain"]:
        is_empty = False

        # NaN 처리
        if pd.isna(value):
            is_empty = True
        else:
            try:
                parsed = ast.literal_eval(str(value))

                # [] 처리
                if parsed == []:
                    is_empty = True

                # [""] 처리
                elif parsed == [""]:
                    is_empty = True

                else:
                    # 리스트 내부 값 count
                    if isinstance(parsed, list):
                        valid_items = []

                        for item in parsed:
                            item = str(item).strip()

                            if item != "":
                                valid_items.append(item)

                        # 한 row 안에서 중복 제거 후 count
                        for item in set(valid_items):
                            label_counter[item] += 1

            except Exception:
                # 파싱 실패 시 문자열 자체 검사
                value_str = str(value).strip()

                if value_str in ["", "[]", '[""]', "['']"]:
                    is_empty = True
                else:
                    label_counter[value_str] += 1

        if is_empty:
            empty_count += 1

# 결과 출력
print(f"비어있는 original_domain 개수: {empty_count}")
print(f"고유 라벨 개수: {len(label_counter)}")

# DataFrame 생성
result_df = pd.DataFrame(
    [
        {
            "original_domain_label": label,
            "count": count
        }
        for label, count in label_counter.items()
    ]
)

# count 기준 내림차순 정렬
result_df = result_df.sort_values(
    by="count",
    ascending=False
).reset_index(drop=True)

# CSV 저장
result_df.to_csv(output_path, index=False)

print(f"고유 라벨 CSV 저장 완료: {output_path}")

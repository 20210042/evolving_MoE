---
annotations_creators: []
language_creators:
- crowdsourced
- expert-generated
language:
- code
license: apache-2.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets: []
task_categories:
- text-generation
task_ids:
- language-modeling
paperswithcode_id: taco-topics-in-algorithmic-code-generation
pretty_name: TACO
tags:
- code
dataset_info:
  config_name: ALL
  features:
  - name: question
    dtype: string
  - name: solutions
    dtype: string
  - name: starter_code
    dtype: string
  - name: input_output
    dtype: string
  - name: difficulty
    dtype: string
  - name: raw_tags
    dtype: string
  - name: name
    dtype: string
  - name: source
    dtype: string
  - name: tags
    dtype: string
  - name: skill_types
    dtype: string
  - name: url
    dtype: string
  - name: Expected Auxiliary Space
    dtype: string
  - name: time_limit
    dtype: string
  - name: date
    dtype: string
  - name: picture_num
    dtype: string
  - name: memory_limit
    dtype: string
  - name: Expected Time Complexity
    dtype: string
  splits:
  - name: train
    num_bytes: 4239311973
    num_examples: 25443
  - name: test
    num_bytes: 481480755
    num_examples: 1000
  download_size: 2419844942
  dataset_size: 4720792728
configs:
- config_name: ALL
  data_files:
  - split: train
    path: ALL/train-*
  - split: test
    path: ALL/test-*
---

# TACO Dataset

<img src="https://cdn-uploads.huggingface.co/production/uploads/6335113375bed9932474315e/rMxdXcC56S3FEh37oRa2s.png" width="200" height="200">

[TACO](https://github.com/FlagOpen/TACO) is a benchmark for code generation with 26443 problems. It can be used to evaluate the ability of language models to generate code from natural language specifications.

## Key Update:
We remove and modified some test cases in test set. Please update to use the newest version.

## Dataset Description

- **Repository:** https://github.com/FlagOpen/TACO/
- **Paper:** [TACO: Topics in Algorithmic COde generation dataset](https://arxiv.org/abs/2312.14852)
- **Leaderboard:** [Code Generation on CodeContests](https://paperswithcode.com/sota/code-generation-on-taco-code)
- **Point of Contact:** [Bo-Wen Zhang](mailto:bwzhang@baai.ac.cn)

## Languages

The dataset contains questions in English and code solutions in Python.

## Dataset Structure

```python
from datasets import load_dataset
load_dataset("BAAI/TACO")

DatasetDict({
    train: Dataset({
        features: ['question', 'solutions', 'starter_code', 'input_output', 'difficulty', 'raw_tags', 'name', 'source', 'tags', 'skill_types', 'url', 'Expected Auxiliary Space', 'time_limit', 'date', 'picture_num', 'memory_limit', 'Expected Time Complexity'],
        num_rows: 25443
    })
    test: Dataset({
        features: ['question', 'solutions', 'starter_code', 'input_output', 'difficulty', 'raw_tags', 'name', 'source', 'tags', 'skill_types', 'url', 'Expected Auxiliary Space', 'time_limit', 'date', 'picture_num', 'memory_limit', 'Expected Time Complexity'],
        num_rows: 1000
    })
})
```

### How to use it

You can load and iterate through the dataset with the following two lines of code for the train split:

```python
from datasets import load_dataset
import json

ds = load_dataset("BAAI/TACO", split="train")
sample = next(iter(ds))
# non-empty solutions and input_output features can be parsed from text format this way:
sample["solutions"] = json.loads(sample["solutions"])
sample["input_output"] = json.loads(sample["input_output"])
sample["raw_tags"] = eval(sample["raw_tags"])
sample["tags"] = eval(sample["tags"])
sample["skill_types"] = eval(sample["skill_types"])
print(sample)

#OUTPUT:
{
  "question": "You have a deck of $n$ cards, and you'd like to reorder it to a new one.\n\nEach card has a value between $1$ and $n$ equal to $p_i$. ...",
  "solutions": [
    "import heapq\nfrom math import sqrt\nimport operator\nimport sys\ninf_var = 0\nif inf_var == 1:\n\tinf = open('input.txt', 'r')\nelse:\n\tinf = sys.stdin\n ...",
    "t = int(input())\nfor _ in range(t):\n\tn = int(input())\n\tp = list(map(int, input().split()))\n\tans = []\n\tp1 = [-1] * (n + 1)\n\tfor i in range(n):\n\t\tp1[p[i]] = i\n\ti = n\n\twhile i:\n\t\twhile i > 0 and p1[i] == -1:\n\t\t\ti -= 1\n\t\telse:\n\t\t\tif i:\n\t\t\t\tk = 0\n\t\t\t\tfor j in range(p1[i], n):\n\t\t\t\t\tans.append(p[j])\n\t\t\t\t\tp1[p[j]] = -1\n\t\t\t\t\tk += 1\n\t\t\t\tn -= k\n\t\t\t\ti -= 1\n\t\t\telse:\n\t\t\t\tbreak\n\tprint(*ans)\n",
    "import sys\n\ndef get_ints():\n\treturn map(int, sys.stdin.readline().strip().split())\n\ndef get_list():\n\treturn list(map(int, sys.stdin.readline().strip().split()))\n\ndef get_list_string():\n\treturn list(map(str, sys.stdin.readline().strip().split()))\n\ndef get_string():\n\treturn sys.stdin.readline().strip()\n\ndef get_int():\n\treturn int(sys.stdin.readline().strip())\n\ndef get_print_int(x):\n\tsys.stdout.write(str(x) + '\\n')\n\ndef get_print(x):\n\tsys.stdout.write(x + '\\n')\n\ndef get_print_int_same(x):\n\tsys.stdout.write(str(x) + ' ')\n\ndef get_print_same(x):\n\tsys.stdout.write(x + ' ')\nfrom sys import maxsize\n\ndef solve():\n\tfor _ in range(get_int()):\n\t\tn = get_int()\n\t\tarr = get_list()\n\t\ti = n - 1\n\t\tj = n - 1\n\t\ttemp = sorted(arr)\n\t\tvis = [False] * n\n\t\tans = []\n\t\twhile j >= 0:\n\t\t\tt = j\n\t\t\ttt = []\n\t\t\twhile t >= 0 and arr[t] != temp[i]:\n\t\t\t\tvis[arr[t] - 1] = True\n\t\t\t\ttt.append(arr[t])\n\t\t\t\tt -= 1\n\t\t\tvis[arr[t] - 1] = True\n\t\t\ttt.append(arr[t])\n\t\t\ttt = tt[::-1]\n\t\t\tfor k in tt:\n\t\t\t\tans.append(k)\n\t\t\tj = t - 1\n\t\t\twhile i >= 0 and vis[i]:\n\t\t\t\ti -= 1\n\t\tget_print(' '.join(map(str, ans)))\nsolve()\n",
    ...
  ],
  "starter_code": "",
  "input_output": {
    "inputs": [
      "4\n4\n1 2 3 4\n5\n1 5 2 4 3\n6\n4 2 5 3 6 1\n1\n1\n",
      "4\n4\n2 1 3 4\n5\n1 5 2 4 3\n6\n4 2 5 3 6 1\n1\n1\n",
      "4\n4\n2 1 3 4\n5\n1 5 2 4 3\n6\n2 4 5 3 6 1\n1\n1\n",
      "4\n4\n1 2 3 4\n5\n1 5 2 4 3\n6\n4 2 5 3 6 1\n1\n1\n"
    ],
    "outputs": [
      "4 3 2 1\n5 2 4 3 1\n6 1 5 3 4 2\n1\n",
      "4 3 2 1\n5 2 4 3 1\n6 1 5 3 4 2\n1\n",
      "4 3 2 1\n5 2 4 3 1\n6 1 5 3 4 2\n1\n",
      "\n4 3 2 1\n5 2 4 3 1\n6 1 5 3 4 2\n1\n"
    ]
  },
  "difficulty": "EASY",
  "raw_tags": [
    "data structures",
    "greedy",
    "math"
  ],
  "name": null,
  "source": "codeforces",
  "tags": [
    "Data structures",
    "Mathematics",
    "Greedy algorithms"
  ],
  "skill_types": [
    "Data structures",
    "Greedy algorithms"
  ],
  "url": "https://codeforces.com/problemset/problem/1492/B",
  "Expected Auxiliary Space": null,
  "time_limit": "1 second",
  "date": "2021-02-23",
  "picture_num": "0",
  "memory_limit": "512 megabytes",
  "Expected Time Complexity": null
}
```
Each sample consists of a programming problem formulation in English, some ground truth Python solutions, test cases that are defined by their inputs and outputs and function name if provided, as well as some metadata regarding the difficulty level (difficulty), topics of task (raw tags), algorithms (tags) as well as required programming skill types (skill_types) of the problem and its source. 

If a sample has non empty `input_output` feature, you can read it as a dictionary with keys `inputs` and `outputs` and `fn_name` if it exists, and similarily you can parse the solutions into a list of solutions as shown in the code above. 

You can also filter the dataset for the difficulty level: EASY, MEDIUM, MEDIUM_HARD, HARD and VERY_HARD, or filter the programming skill types: Amortized analysis, Bit manipulation, Complete search, Data structures, Dynamic programming, Greedy algorithms, Range queries, Sorting. Just pass the list of difficulties or skills as a list. E.g. if you want the most challenging problems, you need to select the VERY_HARD level:

```python
ds = load_dataset("BAAI/TACO", split="train", difficulties=["VERY_HARD"])
print(next(iter(ds))["question"])
```
```
#OUTPUT:
"""Let S(n) denote the number that represents the digits of n in sorted order. For example, S(1) = 1, S(5) = 5, S(50394) = 3459, S(353535) = 333555.
Given a number X, compute <image> modulo 109 + 7.

Input
The first line of input will contain the integer X (1 ≤ X ≤ 10700).

Output
Print a single integer, the answer to the question.

Examples

Input
21

Output
195

Input
345342

Output
390548434

Note

The first few values of S are 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 12. The sum of these values is 195.
```
Or if you want the problems invovled with Range queries and Sorting, you need to select the skills Range queries and Sorting:

```python
ds = load_dataset("BAAI/TACO", split="train", skills=["Range queries", "Sorting"])
```

### Data Fields

|Field|Type|Description|
|---|---|---|
|question|string|problem description|
|solutions|string|some python solutions|
|input_output|string|Json string with "inputs" and "outputs" of the test cases, might also include "fn_name" the name of the function|
|difficulty|string|difficulty level of the problem|
|picture_num|string|the number of pictures in the problem|
|source|string|the source of the problem|
|url|string|url of the source of the problem|
|date|string|the date of the problem|
|starter_code|string|starter code to include in prompts|
|time_limit|string|the time consumption limit to solve the problem|
|memory_limit|string|the memory consumption limit to solve the problem|
|Expected Auxiliary Space|string|the extra auxiliary space expected to solve the problem|
|Expected Time Complexity|string|the time complexity expected to solve the problem|
|raw_tags|string|the topics of the programming task|
|tags|string|the manually annoatated algorithms needed to solve the problem|
|skill_types|string|the mapped programming skill types to solve the problem|



### Data Splits

The dataset contains a train with 25443 samples and test splits with 1000 samples.

### Dataset Statistics
* 26443 coding problems
* 1.55M verified solutions
* for tests split, the average number of test cases is 202.3
* all files have ground-truth solutions in the test split

## Dataset Creation

To create the TACO dataset, the authors manually curated problems from open-access sites where programmers share problems with each other, including Aizu
AtCoder, CodeChef, Codeforces, CodeWars, GeeksforGeeks, HackerEarth, HackerRank, Katti and LeetCode. For more details please refer to the original paper.

## License
The TACO dataset that is authored by BAAI, Shandong Normal University and Peking University is released under an [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). However, the data also includes content licensed under other permissive licenses such as MIT License, or web-crawled data which is used under the terms of the CC BY 4.0 license ([Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/legalcode)).

We gratefully acknowledge the contributions of the following:
*   some AtCoder, Codeforces, CodeWars, Kattis, LeetCode material curated from APPS dataset (https://github.com/hendrycks/apps)
*   some Aizu, AtCoder, CodeChef, Codeforces material curated from CodeContest dataset (https://github.com/google-deepmind/code_contests)
*   Codeforces materials are sourced from http://codeforces.com.
*   CodeChef materials are sourced from https://www.codechef.com.
*   GeekforGeeks materials are sourced from https://www.geeksforgeeks.org
*   HackerEarth materials are curated from:
    [Description2Code Dataset](https://github.com/ethancaballero/description2code),
    licensed under the
    [MIT open source license](https://opensource.org/licenses/MIT), copyright
    not specified.
*   HackerRank materials are sourced from https://www.hackerrank.com. We don't know what the legal rights or data licenses of HackerRank. Please contact us if there is data license.
## Citation Information

If you find our data, or code helpful, please cite [the original paper](https://arxiv.org/abs/2312.14852):

```
@article{li2023taco,
  title={TACO: Topics in Algorithmic COde generation dataset},
  author={Rongao Li and Jie Fu and Bo-Wen Zhang and Tao Huang and Zhihong Sun and Chen Lyu and Guang Liu and Zhi Jin and Ge Li},
  journal={arXiv preprint arXiv:2312.14852},
  year={2023}
}
```
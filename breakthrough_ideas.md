# [기술 사양서] evolving_MoE 성능 최적화 및 Local Optimum 탈출을 위한 아키텍처 개선

본 문서는 `bigmath` 및 고난도 코딩 벤치마크 실험 중 발견된 아키텍처적 병목 현상(특정 도메인의 무한 루프, 에이전트의 조기 퇴출, 컨텍스트 폭발 위험)을 해결하기 위한 세 가지 핵심 기능의 기술 사양을 정의합니다.

## 1. 핵심 변경 사양

### 기능 1: Fractional WAR (공동 기여도 산정 로직 도입)
* **목적:** 복수의 에이전트가 문제를 함께 맞힌 경우 정답 기여도를 나누어 가짐으로써, 에이전트들의 생존력을 안정적으로 보존합니다.
* **대상 파일:** `src/war.py`

```python
def compute_war_scores(
    squad_results: Dict[str, Set[str]],
    total_batch_size: int,
    *,
    tiebreak: str = "random",
    rng: random.Random | None = None,
) -> tuple[Dict[str, float], int, float]:
    if not squad_results:
        return {}, 0, 0.0

    all_solved: Set[str] = set().union(*squad_results.values())
    upper_bound_count = len(all_solved)
    upper_bound_rate = (upper_bound_count / total_batch_size) * 100 if total_batch_size > 0 else 0.0

    problem_to_solvers = {}
    for agent_id, solved_set in squad_results.items():
        for pid in solved_set:
            problem_to_solvers.setdefault(pid, set()).add(agent_id)

    war_scores: Dict[str, float] = {agent_id: 0.0 for agent_id in squad_results.keys()}
    for agent_id, solved_set in squad_results.items():
        for pid in solved_set:
            solvers_count = len(problem_to_solvers[pid])
            war_scores[agent_id] += 1.0 / solvers_count

    return war_scores, upper_bound_count, upper_bound_rate
```

### 기능 2: Hard Error Quarantine (독성 문제 격리 로직)
* **목적:** 모델 체급 한계로 해결이 불가능한 '독성 문제'가 Scout 컨텍스트를 독점하는 현상을 방지하기 위해, 지속해서 실패한 문제를 격리(Quarantine) 처리합니다.
* **대상 파일:** `src/orchestrator.py`

```python
class GMEvolutionOrchestrator:
    def __init__(self, ...):
        self.problem_fail_counts = {}
        self.quarantined_problems = set()
        self.max_fail_threshold = 3

    def run_epoch(self, batch_data: List[Dict[str, Any]]) -> None:
        squad_results, hard_errors_texts = self.run_batch(batch_data)
        
        for pid in hard_errors_texts.keys():
            self.problem_fail_counts[pid] = self.problem_fail_counts.get(pid, 0) + 1
            if self.problem_fail_counts[pid] >= self.max_fail_threshold:
                self.quarantined_problems.add(pid)
                logging.info(f"⚠️ Problem {pid} quarantined due to persistent failures.")

        active_hard_errors = {
            pid: txt for pid, txt in hard_errors_texts.items() 
            if pid not in self.quarantined_problems
        }
        
        if not active_hard_errors:
            logging.info("No active hard errors remaining (or all quarantined).")
            return

        hard_errors_combined = "\n\n---\n\n".join(active_hard_errors.values())
```

### 기능 3: Metadata & Error Type Summary (Scout 컨텍스트 요약화)
* **목적:** 하드 에러의 전체 본문을 덤프하는 대신, 오답 유형과 도메인 정보를 정형화된 통계 및 핵심 샘플 형태로 압축하여 컨텍스트 효율성을 극대화합니다.
* **대상 파일:** `src/scout.py`

```python
def build_structured_hard_error_report(active_hard_errors: List[Dict[str, Any]]) -> str:
    total_errors = len(active_hard_errors)
    
    domain_counts = {}
    error_types = {}
    for err in active_hard_errors:
        domain_counts[err['domain']] = domain_counts.get(err['domain'], 0) + 1
        error_types[err['error_type']] = error_types.get(err['error_type'], 0) + 1

    report = [
        "### Hard Error Distribution Summary",
        f"- Total Persistent Failures: {total_errors}",
        f"- By Sub-domain: {json.dumps(domain_counts)}",
        f"- By Failure Type: {json.dumps(error_types)}",
        "\n### Representative Sample Unsolved Problems"
    ]
    
    samples = active_hard_errors[:3]
    for idx, sample in enumerate(samples):
        report.append(f"\n[Sample {idx+1}] ID: {sample['id']} | Domain: {sample['domain']} | Type: {sample['error_type']}\nDescription:\n{sample['text'][:1000]}")
        
    return "\n".join(report)
```

## 2. 기대 효과
* **Local Optimum 탈출:** 풀 수 없는 문제에 묶여 진화 흐름이 마비되던 병목이 제거됩니다.
* **로스터 내 다양성 및 생존력 유지:** 안정적인 Fractional WAR 방식을 통해 유용한 에이전트들이 연쇄 퇴출되는 스노우볼 현상을 방지합니다.
* **토큰 비용 최적화 및 확장성:** Scout 프롬프트의 토큰 크기가 압축되어 대규모 진화 실험이 가능해집니다.
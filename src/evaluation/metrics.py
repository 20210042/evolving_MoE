import logging
import re
import string
from collections import Counter

from math_verify import parse as _mv_parse, verify as _mv_verify


logger = logging.getLogger(__name__)


# ============================================================================
# EM, token-level f1 score 계산
# ============================================================================

def _normalize_text(text: str, is_finance: bool = False) -> str:
    """
    비교를 위해 텍스트를 정규화한다.
    소문자화 → 구두점 제거 → 공백 정리

    Args:
        text: 원본 텍스트 문자열.

    Returns:
        정규화된 텍스트 문자열.
    """
    if not isinstance(text, str): text = str(text)
    
    text = text.lower().strip()     # 구두점 제거
    
    
    if is_finance:  ## 금융 데이터셋의 경우, .을 제거하면 안됨. 
                    ## TODO: 근데 replace(' ', '') 이건 하는 게 맞나?
        text = text.replace(' ', '').replace(',', '').replace('$', '')
    
    ## .를 포함해서, 다음 특수문자들을 모두 제거하는 코드. !"#$%&'()*+,-./:;<=>?@[\]^_ {|}~
    else: 
        text = text.translate(str.maketrans("", "", string.punctuation))    # 연속 공백을 하나로 
    
    text = " ".join(text.split())
        
    return text
    

def exact_match_score(prediction: str, reference: str, is_finance: bool = False) -> float:
    """
    정규화 후 정확 일치(Exact Match) 점수를 계산한다.

    대소문자, 구두점, 공백을 무시하고 비교.

    Args:
        prediction: 모델이 생성한 답변 (str).
        reference: 정답 (str).

    Returns:
        float: 1.0 (일치) 또는 0.0 (불일치).
    """
    return 1.0 if _normalize_text(prediction, is_finance=is_finance) == _normalize_text(reference, is_finance=is_finance) else 0.0


def f1_score_tokens(prediction: str, reference: str, is_finance: bool = False) -> float:
    """
    토큰 레벨 F1 점수를 계산한다.

    SQuAD 스타일 QA 평가에서 표준적으로 사용되는 방식.
    정규화된 텍스트를 공백으로 토큰화한 후,
    Precision = (공통 토큰 수) / (예측 토큰 수)
    Recall = (공통 토큰 수) / (정답 토큰 수)
    F1 = 2 * Precision * Recall / (Precision + Recall)

    Args:
        prediction: 모델이 생성한 답변 (str).
        reference: 정답 (str).

    Returns:
        float: F1 점수 (0.0 ~ 1.0).
    """
    pred_tokens = _normalize_text(prediction, is_finance=is_finance).split()
    ref_tokens = _normalize_text(reference, is_finance=is_finance).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ============================================================================
# 숫자 비교
# 
# - numerical_match     : 포맷 차이(소수↔%, 콤마, $)와 반올림을 허용하지만, 분수나 LaTeX 형태는 파싱 불가.
# - math_verify_accuracy: LaTeX/분수까지 고려해서 수학적으로 동일한지 판단하지만, 소수를  반올림 오차는 허용하지 않음. 
#
#
#   pred              reference     numerical_match   math_verify   비고
#   "0.64"            "64%"         True              True          둘 다 처리 (0.64 = 64%)
#   "0.6422"          "64%"         True              False         ×100=64.22, 상대오차 0.34% < 1%
#   "64.22%"          "64%"         True              False         상대오차 0.34% < 1% / 64.22 ≠ 64
#   "1235"            "1234.56"     True              False         반올림 오차 0.04%
#   "1/2"             "0.5"         False             True          분수 파싱 불가 vs 수학적 동치
#   "\\frac{1}{2}"    "0.5"         False             True          LaTeX 파싱
#   "hello"           "64"          False             False         파싱 불가
#
# ============================================================================

def numerical_match(prediction: str, reference: str, tolerance: float = 1e-3) -> float:
    """숫자 답변의 일치 여부를 판단한다.

    소수↔퍼센트 변환은 의미적으로 동치이므로 허용, 반올림 오차는 0.1% 이내 허용(1e-3).
      - "0.64"   vs "64%"   → True  (0.64 × 100 = 64, 정확 일치)
      - "64.22%" vs "64%"   → False (상대오차 0.34% > 1e-3)
      - "1.4241" vs "1.4241"→ True
      - "1.4242" vs "1.4241"→ True  (상대오차 7e-5 < 1e-3)
      - "64.2%"  vs "64.22%"→ True  (상대오차 0.03% < 1e-3, 소수점 반올림 허용)

    소수↔퍼센트 변환 조건: abs(소수값) <= 2.0 인 경우만 변환.
      - "64.22" vs "64%": 64.22 > 2.0 이므로 변환하지 않고 직접 비교
      - "0.64"  vs "64%": 0.64 <= 2.0 이므로 ×100 후 비교

    Args:
        prediction: 모델이 생성한 답변.
        reference:  정답.
        tolerance:  허용 상대오차 (기본값: 1e-3). 소수점 반올림 오차를 허용하되 명백히 다른 값은 거름.

    Returns:
        1.0 (일치) 또는 0.0 (불일치 또는 파싱 실패).
    """
    
    def _parse_number(s: str) -> tuple[float | None, bool]:
        """숫자 문자열을 정규화 및 % 여부 기록"""
        s = str(s).strip()
        is_percent = "%" in s  # % 기호는 제거 전에 먼저 기록
        
        ## 숫자 정규화
        cleaned = s.replace(",", "").replace("$", "").replace("%", "").strip()
        
        try:
            return float(cleaned), is_percent
        except ValueError:
            return None, is_percent

    
    ## 숫자 정규화, %표기 여부 기록. --> 파싱 실패하면 틀린 것으로 간주한다.
    pred_val, pred_is_percent = _parse_number(prediction)
    ref_val, ref_is_percent = _parse_number(reference)
    if pred_val is None or ref_val is None: return 0.0  


    ## 소수↔퍼센트 스케일 맞추기. 
    ## 정답이 %인데 예측이 소수이고, 예측값이 2 이하면 ×100해서 백분율로 스케일을 맞춘다. 
    if ref_is_percent and not pred_is_percent and abs(pred_val) <= 2.0: pred_val *= 100.0
    elif pred_is_percent and not ref_is_percent and abs(ref_val) <= 2.0: ref_val *= 100.0

    
    ## 정답이 0이면, 상대오차의 분모가 0이 되어 계산이 안되므로, 절대오차로 비교.
    ## 그 외의 경우에는, 상대오차로 비교
    if abs(ref_val) < 1e-9: 
        return 1.0 if abs(pred_val - ref_val) < tolerance else 0.0
    else:
        return 1.0 if abs(pred_val - ref_val) / abs(ref_val) < tolerance else 0.0



def math_verify_accuracy(prediction: str, reference: str) -> float:
    """
    math_verify 기반 정답 검증.
    - reference가 파싱 가능하면 수학 검증 결과(1/0)를 반환한다.
    - reference가 파싱 불가한 경우, 0.0을 반환한다.
    """
    ## math-verfiy 형식으로 숫자 파싱. '64%' → [64*(1/100), '64']
    ## 파싱 실패시 틀린 것으로 간주하고 0.0 반환
    ref = _mv_parse(str(reference))
    pred = _mv_parse(str(prediction))
    if not ref or not pred: return 0.0

    return 1.0 if _mv_verify(pred, ref) else 0.0


# ============================================================================
# 모델 출력에서 최종 답 추출
# ============================================================================

def extract_answer(text: str) -> str:
    """모델 출력에서 최종 답을 추출한다.

    1순위: 텍스트 내 마지막 "Final Answer: ..." 라인의 내용.
    2순위: 텍스트 내 마지막 \\boxed{...} 의 내용 (중첩 괄호 지원).
    3순위: 원본 텍스트 전체를 그대로 반환.

    Args:
        text: 모델이 생성한 응답 전체.

    Returns:
        추출된 답 문자열.
    """
    # 1) "Final Answer: X"
    fa_matches = re.findall(r"Final Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if fa_matches:
        return fa_matches[-1].strip()

    # 2) \boxed{} — 중첩 괄호 지원 (e.g. \boxed{\frac{1}{2}})
    boxed: list[str] = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        start = idx + len(r"\boxed{")
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            boxed.append(text[start : j - 1])
        i = idx + 1
    if boxed:
        return boxed[-1].strip()

    # 3) raw text
    return text.strip()
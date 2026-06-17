import logging
import re
import string
from collections import Counter

from math_verify import parse as _mv_parse, verify as _mv_verify

try:  
    from math_verify import LatexExtractionConfig as _LatexCfg, ExprExtractionConfig as _ExprCfg
except Exception:  
    from math_verify.parser import LatexExtractionConfig as _LatexCfg, ExprExtractionConfig as _ExprCfg

# math_verify.parse는 $...$ 나 \boxed{} 로 감싼 내용에만 LaTeX extractor가 동작
# Raw LaTeX(예: "\frac{5\pi}{12}", "\dfrac{1}{6}")는 []로 파싱돼 오답으로 처리
# → 답이 감싸지지 않은 분수/기호인 문제는 예측이 정답과 글자까지 같아도 자동 오답
_MV_EXTRACTION = [_LatexCfg(), _ExprCfg()]


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


def token_f1_score(prediction: str, reference: str, is_finance: bool = False) -> float:
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

def numerical_match_score(prediction: str, reference: str, tolerance: float = 1e-3) -> float:
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



def _mv_wrap(text: str) -> str:
    """Raw LaTeX를 $...$ 로 감싸 math_verify의 LaTeX extractor가 동작하게 한다.
    이미 $ 나 \\boxed 가 있으면 그대로 둔다."""
    s = str(text).strip()
    if "$" in s or "\\boxed" in s:
        return s
    return f"${s}$"


def math_verify_score(prediction: str, reference: str) -> float:
    """
    math_verify 기반 정답 검증.
    - reference가 파싱 가능하면 수학 검증 결과(1/0)를 반환한다.
    - reference가 파싱 불가한 경우, 0.0을 반환한다.
    """
    ## math-verfiy 형식으로 파싱. '64%' → [64*(1/100), '64']
    ## Raw LaTeX도 잡도록 $로 감싸기 + Latex/Expr extractor 명시. 파싱 실패 시 0.0.
    ref = _mv_parse(_mv_wrap(reference), extraction_config=_MV_EXTRACTION)
    pred = _mv_parse(_mv_wrap(prediction), extraction_config=_MV_EXTRACTION)
    if not ref or not pred: return 0.0

    ## 표현 순서/형식 차이(예: ap+p^2 vs p^2+ap)를 잡기 위해 양방향 검증
    return 1.0 if (_mv_verify(ref, pred) or _mv_verify(pred, ref)) else 0.0


# --- 객관식(MC) 고려 채점 -------------------------------------------------------
# NuminaMath 객관식은 gold가 보기 문자("B")와 값("a+b = 3")으로 섞여 있어, 모델이
# 한 포맷으로 박싱하면 절반은 실력과 무관하게 FAIL한다. mc_aware_math_score()는
# math_verify_score 위에 얹어서 객관식이면 모델 boxed 부분이 정답 보기의 문자 또는
# 값과 맞아도 정답 인정. 객관식 아니면 math_verify_score와 동일.
_MC_OPT_RE = re.compile(r"(?:^|\n|\s)\(?([A-D])[\):.]\s*(.+?)(?=(?:\n|\s)\(?[A-D][\):.]|\Z)", re.S)


def _parse_mc_options(instruction: str) -> dict:
    opts: dict = {}
    for m in _MC_OPT_RE.finditer(instruction or ""):
        opts.setdefault(m.group(1), m.group(2).strip()[:80])
    return opts


def _mc_correct_letter(gold: str, opts: dict):
    g = (gold or "").strip()
    for pat in (r"^\(?([A-D])[\):.\s]", r"^\\?text\{?\s*\(?([A-D])", r"^([A-D])$"):
        m = re.match(pat, g)
        if m:
            return m.group(1)
    for L, v in opts.items():  # gold가 값 → 값이 일치하는 보기 찾기
        try:
            if math_verify_score(v, gold):
                return L
        except Exception:
            pass
    return None


def mc_score(prediction: str, reference: str, instruction: str = "") -> float:
    """math_verify_score + 객관식 보기↔값 동치 인정 (추가 전용)."""
    
    opts = _parse_mc_options(instruction)
    if len(opts) < 3:  # 객관식 아님 → math_verify_score와 동일
        return 0.0
    
    cl = _mc_correct_letter(str(reference or ""), opts)
    if not cl:
        return 0.0
    
    p = (prediction or "").strip()
    if re.fullmatch(r"\(?([A-D])\)?", p) and re.sub(r"[^A-D]", "", p) == cl:
        return 1.0  # 모델이 정답 보기 문자 박싱
    try:
        if math_verify_score(p, opts.get(cl, "")):
            return 1.0  # 모델이 정답 보기 값 박싱
        
    except Exception:
        pass
    return 0.0
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
# NuminaMath 객관식은 gold가 보기 문자("B", "\textbf{(C)}")와 실제 값("a+b = 3",
# "\pi")으로 섞여 있다. 모델도 보기 문자만 쓰거나, 값만 쓰거나, "C: value"처럼
# 둘을 같이 쓰기 때문에 한 포맷만 비교하면 맞는 답을 놓치기 쉽다.
#
# mc_score()의 원칙:
# 1. 문제 본문에서 A-E 선택지와 각 선택지 값을 파싱한다.
# 2. gold가 보기 문자/문자+값이면 정답 문자를 바로 얻고, gold가 값만 있으면
#    선택지 값들과 비교해서 정답 문자를 추론한다.
# 3. prediction이 "C"처럼 문자만 있으면 정답 문자와 같을 때 인정한다.
# 4. prediction이 "C: value"처럼 문자와 값을 같이 갖고 있으면, 문자가 맞아야 하며
#    value도 정답 선택지 값 또는 gold value와 모순되지 않아야 인정한다. 이 조건은
#    "C"는 맞지만 뒤에 붙인 값이 틀린 false positive를 막기 위한 AND 조건이다.
# 5. prediction이 값만 있으면 정답 선택지 값 또는 gold와 동치일 때 인정한다.
#
# 값 비교는 빠른 exact/텍스트 정규화 비교를 먼저 하고, 마지막에 math_verify_score()를
# 사용한다. 이렇게 해서 LaTeX spacing, \textbf{(C)}, \boxed{}, π/\pi 같은 포맷 차이는
# 넓게 허용하되, 명백히 다른 선택지나 다른 값은 보수적으로 오답 처리한다.




# 선택지 marker만 잡기 위한 정규식 조각.
# 일반 문장의 "A ..."를 보기로 오인하지 않도록 "A:", "A)", "(A)",
# "\textbf{(A)}"처럼 명확한 marker만 허용한다.
_MC_MARKER_RE = (
    r"(?:"
    r"\\(?:textbf|mathbf|mathrm|text)\{\s*\(?([A-E])\)?\s*\}"
    r"|\(([A-E])\)"
    r"|([A-E])[\):.]"
    r")"
)

# instruction 전체에서 객관식 선택지와 값을 순서대로 추출한다.
# 예: "A: $1$ B: $2$" 또는 "\textbf{(A)}\ 1\qquad\textbf{(B)}\ 2".
# 마지막 캡처 그룹이 해당 선택지의 value이고, 다음 marker 전까지 non-greedy로 잡는다.
_MC_OPT_RE = re.compile(
    r"(?:^|\n|\s|\\qquad|\\quad)"
    r"(?:\$?\s*)?"
    + _MC_MARKER_RE
    + r"(?:\$?\s*)?"
    + r"\\?\s*"
    + r"(.+?)"
    + r"(?="
    + r"(?:\n|\s|\\qquad|\\quad)"
    + r"(?:\$?\s*)?"
    + _MC_MARKER_RE
    + r"(?:\$?\s*)?"
    + r"\\?\s*"
    + r"|\Z)",
    re.S,
)


def _parse_mc_options(instruction: str) -> dict:
    """문제 본문에서 객관식 선택지를 {letter: value} 형태로 파싱한다."""
    opts: dict = {}
    for m in _MC_OPT_RE.finditer(instruction or ""):
        letter = next(g for g in m.groups()[:3] if g)
        value = m.group(4).strip()
        value = re.sub(r"^\\+\s*", "", value).strip()
        value = value.strip("$").strip()
        opts.setdefault(letter, value)
    return opts


def _mc_correct_letter(gold: str, opts: dict):
    """gold answer가 가리키는 정답 보기 문자를 찾는다.

    gold가 "C" 또는 "C: value"면 C를 바로 반환한다. gold가 value-only이면
    각 선택지 value와 비교해서 어떤 보기가 gold와 동치인지 찾는다.
    """
    g = (gold or "").strip()
    letter, _ = _split_mc_letter_value(g)
    if letter:
        return letter
    for L, v in opts.items():  # gold가 값 → 값이 일치하는 보기 찾기
        try:
            if _answers_equivalent(v, gold):
                return L
        except Exception:
            pass
    return None


def _strip_mc_wrapper(text: str) -> str:
    """비교 전에 바깥쪽 $...$ 또는 \boxed{...} wrapper를 제거한다."""
    s = str(text or "").strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()

    boxed = re.fullmatch(r"\\boxed\{(.+)\}", s)
    if boxed:
        s = boxed.group(1).strip()
    return s


def _split_mc_letter_value(text: str) -> tuple[str | None, str]:
    """답변을 (보기 문자, 나머지 값)으로 분리한다.

    "C", "(C)", "C: 3", "\\textbf{(C)}\\ 3" 등을 처리한다.
    보기 문자가 없으면 (None, 원문)을 반환해서 value-only 답으로 취급한다.
    """
    s = _strip_mc_wrapper(text)
    marker = re.match(
        r"^\s*"
        r"(?:\\(?:textbf|mathbf|mathrm|text|operatorname)\{\s*)?"
        r"\(?([A-E])\)?"
        r"(?:\s*\})?"
        r"\s*(?:[:.)])?\s*"
        r"(.*)$",
        s,
        re.S,
    )
    if not marker:
        return None, s

    tail = marker.group(2).strip()
    tail = re.sub(r"^\\+\s*", "", tail).strip()
    return marker.group(1), tail


def _normalize_for_text_compare(text: str) -> str:
    """LaTeX/공백 표기 차이를 줄인 가벼운 문자열 비교용 정규화.

    math_verify를 호출하기 전에 빠르게 동치 판정을 하기 위한 보조 비교다.
    수학적 simplify는 하지 않고, \\textbf{}, \\left/\\right, \\quad, π/\\pi 등
    흔한 포맷 차이만 제거한다.
    """
    s = _strip_mc_wrapper(text)
    s = re.sub(r"\\(?:textbf|mathbf|mathrm|text|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right)", "", s)
    s = re.sub(r"\\(?:quad|qquad|[,;:!])", " ", s)
    s = s.replace("\\ ", " ")
    s = s.replace("π", "pi").replace("\\pi", "pi")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s.lower())
    return s.strip()


def _answers_equivalent(prediction: str, reference: str) -> bool:
    """두 답변 문자열이 같은 답으로 볼 수 있는지 단계적으로 비교한다.

    1) 기존 exact_match_score, 2) MC 전용 텍스트 정규화 비교, 3) math_verify
    순서로 시도한다. math_verify는 느리거나 timeout이 날 수 있어 마지막에 둔다.
    """
    if not str(prediction or "").strip() or not str(reference or "").strip():
        return False
    if exact_match_score(prediction, reference):
        return True
    if _normalize_for_text_compare(prediction) == _normalize_for_text_compare(reference):
        return True
    try:
        return bool(math_verify_score(prediction, reference))
    except Exception:
        return False


def mc_score(prediction: str, reference: str, instruction: str = "") -> float:
    """객관식 보기 문자와 보기 값을 함께 고려해서 정답 여부를 판단한다.

    답이 ``C``처럼 보기 문자만 있으면 문자 일치로 인정한다. ``C: value``처럼
    문자와 값이 같이 있으면 문자가 맞고, 값도 정답 보기/정답과 모순되지 않아야
    인정한다. 답이 값만 있으면 정답 보기 값 또는 gold와 동치일 때 인정한다.
    """
    
    opts = _parse_mc_options(instruction)
    if len(opts) < 3:  # 객관식 아님 → math_verify_score와 동일
        return 0.0
    
    cl = _mc_correct_letter(str(reference or ""), opts)
    if not cl:
        return 0.0
    
    pred_letter, pred_value = _split_mc_letter_value(prediction)
    gold_value = _split_mc_letter_value(reference)[1]
    correct_value = opts.get(cl, "")

    if pred_letter:
        if pred_letter != cl:
            return 0.0
        if not pred_value:
            return 1.0
        if _answers_equivalent(pred_value, correct_value) or _answers_equivalent(pred_value, gold_value):
            return 1.0
        return 0.0

    if _answers_equivalent(prediction, correct_value) or _answers_equivalent(prediction, reference):
        return 1.0
    return 0.0

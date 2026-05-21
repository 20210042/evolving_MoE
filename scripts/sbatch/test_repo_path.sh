#!/bin/bash
# REPO 경로 계산 과정을 단계별로 출력하는 테스트 스크립트

echo "=== REPO 경로 계산 단계별 확인 ==="
echo ""

echo "① \$0 (스크립트 경로):"
echo "   $0"
echo ""

echo "② dirname \"\$0\" (디렉토리만):"
echo "   $(dirname "$0")"
echo ""

echo "③ dirname + /../.. (두 단계 위):"
echo "   $(dirname "$0")/../.."
echo ""

echo "④ cd + pwd (절대경로로 변환):"
echo "   $(cd "$(dirname "$0")/../.." && pwd)"
echo ""

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
echo "⑤ 최종 REPO:"
echo "   $REPO"
echo ""

echo "=== 레포 루트 파일 목록 확인 ==="
ls "$REPO"

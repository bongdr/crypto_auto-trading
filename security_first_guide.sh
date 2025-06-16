# 🔒 보안 우선 설정 가이드
# Git 설정하기 BEFORE 이것부터 먼저 실행하세요!

# ===========================================
# 1단계: .gitignore 파일 생성 (가장 먼저!)
# ===========================================

# 현재 프로젝트 디렉토리에서 실행
# Cursor 터미널에서: Terminal → New Terminal

# .gitignore 파일 생성
cat > .gitignore << 'EOF'
# 🚨 민감한 정보 보호 (API 키 등)
.env
.env.local
.env.*.local
.env.backup

# Python 관련
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
*.egg-info/
dist/
build/

# 로그 파일 (개인정보 포함 가능)
logs/
*.log

# 데이터 캐시 (거래 기록 등)
data_cache/
saved_models/
*.pkl
*.joblib

# 백테스트 결과 (개인 거래 기록)
backtest_results/
results/

# IDE 관련
.vscode/
.idea/
*.swp
*.swo
*~

# macOS
.DS_Store

# 임시 파일
*.tmp
*.temp
temp/
EOF

echo "✅ .gitignore 파일 생성 완료"

# ===========================================
# 2단계: .env 파일 안전하게 백업하기
# ===========================================

# 기존 .env 파일이 있다면 백업
if [ -f ".env" ]; then
    cp .env .env.backup
    echo "✅ 기존 .env 파일을 .env.backup으로 백업했습니다"
else
    echo "ℹ️ .env 파일이 없습니다"
fi

# ===========================================
# 3단계: .env 템플릿 파일 생성
# ===========================================

# GitHub에 올릴 수 있는 템플릿 파일 (실제 키 없음)
cat > .env.template << 'EOF'
# 업비트 API 키 (실제 키로 교체하세요)
UPBIT_ACCESS_KEY=your_actual_access_key_here
UPBIT_SECRET_KEY=your_actual_secret_key_here

# 텔레그램 알림 설정 (선택사항)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# 거래 설정
TRADING_MODE=paper
INITIAL_BALANCE=20000000

# 로깅 설정
LOG_LEVEL=INFO

# 리스크 관리
STOP_LOSS_THRESHOLD=0.05
TAKE_PROFIT_THRESHOLD=0.15
MAX_POSITION_SIZE=0.3

# 데이터 품질 기준
MIN_DATA_DAYS=90
MIN_VOLUME_KRW=10000000000

# ML 모델 설정
MIN_TRAINING_SAMPLES=200
MODEL_RETRAIN_HOURS=24

# 거래 주기 (초)
TRADING_INTERVAL=3600
STATUS_CHECK_INTERVAL=1800
EOF

echo "✅ .env.template 파일 생성 완료 (GitHub에 올라갈 파일)"

# ===========================================
# 4단계: 보안 확인 스크립트 생성
# ===========================================

cat > check_security.py << 'EOF'
#!/usr/bin/env python3
"""
보안 상태 확인 스크립트
"""

import os
import glob

def check_env_files():
    """환경 파일 확인"""
    print("🔍 환경 파일 확인 중...")
    
    # .env 파일들 찾기
    env_files = glob.glob('.env*')
    
    for file in env_files:
        print(f"\n📄 {file}:")
        if file == '.env':
            print("  ⚠️  이 파일은 GitHub에 올라가면 안됩니다!")
        elif file == '.env.template':
            print("  ✅ 템플릿 파일 - GitHub 업로드 OK")
        else:
            print(f"  ❓ 확인 필요한 파일")
        
        # 파일 내용 일부 확인 (키가 실제로 있는지)
        try:
            with open(file, 'r') as f:
                content = f.read()
                if 'your_actual_access_key_here' in content:
                    print("  ✅ 템플릿 내용 - 안전함")
                elif any(key in content for key in ['UPBIT_ACCESS_KEY=', 'TELEGRAM_BOT_TOKEN=']):
                    if any(len(line.split('=')[1].strip()) > 20 for line in content.split('\n') if '=' in line):
                        print("  🚨 실제 API 키가 있을 수 있음 - GitHub 업로드 금지!")
                    else:
                        print("  ✅ 빈 값 또는 템플릿 - 안전함")
        except Exception as e:
            print(f"  ❌ 파일 읽기 오류: {e}")

def check_gitignore():
    """gitignore 확인"""
    print("\n🔍 .gitignore 확인 중...")
    
    if not os.path.exists('.gitignore'):
        print("❌ .gitignore 파일이 없습니다!")
        return False
    
    with open('.gitignore', 'r') as f:
        content = f.read()
    
    required_items = ['.env', '__pycache__/', 'logs/', 'data_cache/']
    
    for item in required_items:
        if item in content:
            print(f"  ✅ {item} - 보호됨")
        else:
            print(f"  ❌ {item} - 누락! .gitignore에 추가 필요")

def main():
    print("🔒 보안 상태 확인")
    print("=" * 50)
    
    check_env_files()
    check_gitignore()
    
    print("\n📋 권장사항:")
    print("1. .env 파일은 절대 Git에 추가하지 마세요")
    print("2. .env.template만 GitHub에 올리세요")
    print("3. 실제 API 키는 로컬에서만 사용하세요")

if __name__ == "__main__":
    main()
EOF

echo "✅ 보안 확인 스크립트 생성 완료"

# ===========================================
# 5단계: 보안 상태 확인 실행
# ===========================================

echo ""
echo "🔍 현재 보안 상태 확인 중..."
python check_security.py

echo ""
echo "✅ 보안 설정 완료!"
echo "이제 안전하게 Git 설정을 시작할 수 있습니다."
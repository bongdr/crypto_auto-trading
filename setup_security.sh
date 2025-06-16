#!/bin/bash
# 간단한 보안 설정 스크립트 - setup_security.sh

echo "🔒 보안 설정 시작..."

# 1. .gitignore 파일 생성
echo "📝 .gitignore 파일 생성 중..."
cat > .gitignore << 'EOF'
# 민감한 정보 보호
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

# 로그 파일
logs/
*.log

# 데이터 캐시
data_cache/
saved_models/
*.pkl
*.joblib

# 백테스트 결과
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

# 2. 기존 .env 파일 백업
if [ -f ".env" ]; then
    cp .env .env.backup
    echo "✅ 기존 .env 파일을 .env.backup으로 백업"
else
    echo "ℹ️ .env 파일이 없습니다"
fi

# 3. .env 템플릿 생성
echo "📝 .env 템플릿 파일 생성 중..."
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

echo "✅ .env.template 파일 생성 완료"

# 4. 보안 확인 스크립트 생성
echo "📝 보안 확인 스크립트 생성 중..."
cat > check_security.py << 'EOF'
#!/usr/bin/env python3
import os
import glob

def check_env_files():
    print("🔍 환경 파일 확인 중...")
    env_files = glob.glob('.env*')
    
    if not env_files:
        print("  ℹ️ .env 파일이 없습니다")
        return
    
    for file in env_files:
        print(f"\n📄 {file}:")
        if file == '.env':
            print("  ⚠️  이 파일은 GitHub에 올라가면 안됩니다!")
        elif file == '.env.template':
            print("  ✅ 템플릿 파일 - GitHub 업로드 OK")
        elif file == '.env.backup':
            print("  ✅ 백업 파일 - 로컬에서만 보관")
        
        try:
            with open(file, 'r') as f:
                content = f.read()
                if 'your_actual_access_key_here' in content or 
'your_bot_token_here' in content:
                    print("  ✅ 템플릿 내용 - 안전함")
                else:
                    lines = [line for line in content.split('\n') if '=' 
in line and not line.startswith('#')]
                    has_real_keys = any(len(line.split('=')[1].strip()) > 
20 for line in lines)
                    if has_real_keys:
                        print("  🚨 실제 API 키가 있을 수 있음 - GitHub 
업로드 금지!")
                    else:
                        print("  ✅ 빈 값 또는 템플릿 - 안전함")
        except Exception as e:
            print(f"  ❌ 파일 읽기 오류: {e}")

def check_gitignore():
    print("\n🔍 .gitignore 확인 중...")
    
    if not os.path.exists('.gitignore'):
        print("❌ .gitignore 파일이 없습니다!")
        return False
    
    with open('.gitignore', 'r') as f:
        content = f.read()
    
    required_items = ['.env', '__pycache__/', 'logs/', 'data_cache/']
    all_protected = True
    
    for item in required_items:
        if item in content:
            print(f"  ✅ {item} - 보호됨")
        else:
            print(f"  ❌ {item} - 누락!")
            all_protected = False
    
    return all_protected

def main():
    print("🔒 보안 상태 확인")
    print("=" * 50)
    
    check_env_files()
    gitignore_ok = check_gitignore()
    
    print("\n📋 결과:")
    if gitignore_ok:
        print("✅ 보안 설정이 올바르게 되어 있습니다!")
        print("이제 Git 설정을 시작할 수 있습니다.")
    else:
        print("❌ 보안 설정에 문제가 있습니다. 다시 확인해주세요.")
    
    print("\n⚠️ 중요:")
    print("- .env 파일은 절대 Git에 추가하지 마세요")
    print("- .env.template만 GitHub에 올리세요")

if __name__ == "__main__":
    main()
EOF

echo "✅ 보안 확인 스크립트 생성 완료"

# 5. 보안 상태 확인
echo ""
echo "🔍 현재 보안 상태 확인 중..."
python3 check_security.py

echo ""
echo "🎉 보안 설정 완료!"
echo "이제 안전하게 Git 설정을 시작할 수 있습니다."
echo ""
echo "다음 단계:"
echo "1. git init"
echo "2. git config --global user.name \"Your Name\""
echo "3. git config --global user.email \"your.email@example.com\""
echo "4. git add ."
echo "5. git commit -m \"Initial commit\""

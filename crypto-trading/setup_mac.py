#!/usr/bin/env python3
"""
Mac 환경 자동 설정 스크립트
"""
import os
import sys
import subprocess

def create_requirements_txt():
    """requirements.txt 파일 생성"""
    requirements = """pyupbit==0.2.31
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
joblib>=1.1.0
requests>=2.28.0
beautifulsoup4>=4.11.0
nltk>=3.7
schedule>=1.1.0
python-dotenv>=0.19.0"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ requirements.txt 파일이 생성되었습니다.")

def create_env_file():
    """.env 파일 템플릿 생성"""
    env_template = """# 업비트 API 키 (실제 키로 교체하세요)
UPBIT_ACCESS_KEY=your_access_key_here
UPBIT_SECRET_KEY=your_secret_key_here

# 텔레그램 설정 (선택사항)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# 거래 설정
TRADING_MODE=paper
LOG_LEVEL=INFO

# 리스크 관리 설정
STOP_LOSS_THRESHOLD=0.05
TAKE_PROFIT_THRESHOLD=0.1
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_template)
        print("✅ .env 파일 템플릿이 생성되었습니다.")
        print("⚠️  .env 파일에 실제 API 키를 입력하세요!")
    else:
        print("ℹ️  .env 파일이 이미 존재합니다.")

def create_directories():
    """프로젝트 디렉토리 구조 생성"""
    directories = [
        'config',
        'data', 
        'models',
        'strategy',
        'trading',
        'sentiment',
        'finance',
        'utils',
        'logs',
        'data_cache',
        'data_cache/sentiment_strategy',
        'data_cache/fund_manager',
        'data_cache/rebalancer', 
        'saved_models',
        'saved_models/sentiment',
        'backtest_results',
        'backtest_results/optimization_results',
        'results'
    ]
    
    created_count = 0
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created_count += 1
        
        # __init__.py 파일 생성 (Python 패키지용)
        package_dirs = ['config', 'data', 'models', 'strategy', 'trading', 'sentiment', 'finance', 'utils']
        if directory in package_dirs:
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write(f'# {directory} 패키지 초기화 파일\n')
    
    print(f"✅ {created_count}개 디렉토리가 생성되었습니다.")

def create_fixed_settings():
    """수정된 settings.py 파일 생성"""
    settings_content = '''import os
from dotenv import load_dotenv
from datetime import datetime

# 환경 변수 로드
load_dotenv()

# 프로젝트 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
BACKTEST_DIR = os.path.join(BASE_DIR, 'backtest_results')

# 로깅 설정
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# 트레이딩 설정
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')

# 업비트 API 키 설정 (환경 변수에서 로드)
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')

# 리스크 관리 설정
STOP_LOSS_THRESHOLD = float(os.getenv('STOP_LOSS_THRESHOLD', '0.05'))
TAKE_PROFIT_THRESHOLD = float(os.getenv('TAKE_PROFIT_THRESHOLD', '0.1'))
TRAILING_STOP_ACTIVATION = 0.05
TRAILING_STOP_DISTANCE = 0.03

# 백테스트 설정
BACKTEST_START_DATE = '2023-01-01'
BACKTEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
BACKTEST_INITIAL_BALANCE = 20000000
BACKTEST_COMMISION = 0.0005

# 데이터 설정
DEFAULT_TIMEFRAME = 'minute30'
DATA_CACHE_EXPIRY = 3600

# 모델 설정
MODEL_VERSION = '1.0.0'
ML_FEATURES = [
    'ma_ratio_5_20', 'ma_ratio_5_60', 'price_to_ma20', 
    'rsi', 'bb_position', 'macd_hist', 'volume_ratio'
]
'''
    
    os.makedirs('config', exist_ok=True)
    with open('config/settings.py', 'w') as f:
        f.write(settings_content)
    print("✅ 수정된 config/settings.py 파일이 생성되었습니다.")

def main():
    """메인 설정 함수"""
    print("🚀 Mac 환경용 가상화폐 자동거래 시스템 설정을 시작합니다...\n")
    
    # 1. requirements.txt 생성
    create_requirements_txt()
    
    # 2. 디렉토리 구조 생성
    create_directories()
    
    # 3. .env 파일 생성
    create_env_file()
    
    # 4. 수정된 settings.py 생성
    create_fixed_settings()
    
    print("\n" + "="*60)
    print("🎉 기본 설정이 완료되었습니다!")
    print("="*60)
    
    print("\n📋 다음 단계:")
    print("1. 패키지 설치: pip install -r requirements.txt")
    print("2. API 키 설정: .env 파일 편집")
    print("3. 연결 테스트: python test_connection.py")

if __name__ == "__main__":
    main()

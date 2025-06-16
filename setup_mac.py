#!/usr/bin/env python3
"""
Mac 환경 자동 설정 스크립트
파일명: setup_mac.py
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
UPBIT_ACCESS_KEY=hf0nxaYZb5OVV1bxO3bpQskVsNENft939azTagYu
UPBIT_SECRET_KEY=SitgryqdwGm2xraQZfefeut5Sxjjs49g64Akuunc

# 텔레그램 설정 (선택사항)
TELEGRAM_BOT_TOKEN=1521613535:AAGPvDTqRLX_ZMxSdQaw4MbDWvkH_ye_xhM
TELEGRAM_CHAT_ID=50134103

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

def install_packages():
    """필요한 패키지 설치"""
    try:
        print("📦 패키지 설치를 시작합니다...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ 기본 패키지 설치가 완료되었습니다.")
        
        # NLTK 데이터 다운로드
        try:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            print("✅ NLTK 데이터 다운로드가 완료되었습니다.")
        except Exception as e:
            print(f"⚠️  NLTK 데이터 다운로드 실패: {e}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False
    return True

def create_simple_test_script():
    """간단한 테스트 스크립트 생성"""
    test_script = '''#!/usr/bin/env python3
"""
간단한 연결 테스트 스크립트
파일명: test_connection.py
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """필수 라이브러리 임포트 테스트"""
    try:
        import pandas as pd
        import numpy as np
        import pyupbit
        print("✅ 기본 라이브러리 임포트 성공")
        return True
    except ImportError as e:
        print(f"❌ 라이브러리 임포트 실패: {e}")
        return False

def test_upbit_connection():
    """업비트 API 연결 테스트"""
    try:
        import pyupbit
        # 공개 API 테스트 (API 키 불필요)
        tickers = pyupbit.get_tickers(fiat="KRW")
        if tickers and len(tickers) > 0:
            print(f"✅ 업비트 연결 성공 ({len(tickers)}개 코인 조회)")
            return True
        else:
            print("❌ 업비트 연결 실패: 데이터 없음")
            return False
    except Exception as e:
        print(f"❌ 업비트 연결 실패: {e}")
        return False

def test_data_collection():
    """데이터 수집 테스트"""
    try:
        import pyupbit
        # 비트코인 일봉 데이터 테스트
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=5)
        if df is not None and len(df) > 0:
            print(f"✅ 데이터 수집 성공 ({len(df)}일 데이터)")
            print(f"   최근 비트코인 가격: {df['close'].iloc[-1]:,.0f}원")
            return True
        else:
            print("❌ 데이터 수집 실패")
            return False
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        return False

if __name__ == "__main__":
    print("🔍 시스템 연결 테스트를 시작합니다...\\n")
    
    tests = [
        ("라이브러리 임포트", test_imports),
        ("업비트 API 연결", test_upbit_connection), 
        ("데이터 수집", test_data_collection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"테스트 중: {test_name}")
        result = test_func()
        results.append(result)
        print()
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"🎯 테스트 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 테스트가 성공했습니다! 거래 시스템을 실행할 준비가 되었습니다.")
        print("\\n다음 명령어로 페이퍼 트레이딩을 시작하세요:")
        print("python main.py --mode paper --balance 1000000")
    else:
        print("⚠️  일부 테스트가 실패했습니다. 설정을 확인해주세요.")
'''
    
    with open('test_connection.py', 'w') as f:
        f.write(test_script)
    print("✅ 테스트 스크립트 (test_connection.py)가 생성되었습니다.")

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
    
    # 5. 테스트 스크립트 생성
    create_simple_test_script()
    
    print("\n" + "="*60)
    print("🎉 기본 설정이 완료되었습니다!")
    print("="*60)
    
    print("\n📋 다음 단계:")
    print("1. 패키지 설치: pip install -r requirements.txt")
    print("2. API 키 설정: .env 파일 편집")
    print("3. 연결 테스트: python test_connection.py")
    print("4. 거래 시작: python main.py --mode paper")
    
    # 패키지 설치 여부 묻기
    install_now = input("\n지금 패키지를 설치하시겠습니까? (y/n): ").lower().strip()
    if install_now in ['y', 'yes']:
        if install_packages():
            print("\n🎉 설정이 완전히 완료되었습니다!")
            print("이제 .env 파일에 API 키를 입력하고 python test_connection.py 로 테스트하세요.")
        else:
            print("\n⚠️  패키지 설치에 실패했습니다. 수동으로 설치해주세요:")
            print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()

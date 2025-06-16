"""
가상화폐 자동거래 시스템 - Mac 환경 설정
"""
import os
import subprocess
import sys

def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        'config', 'data', 'models', 'strategy', 'trading', 
        'sentiment', 'finance', 'utils', 'logs', 'data_cache',
        'data_cache/sentiment_strategy', 'data_cache/fund_manager',
        'data_cache/rebalancer', 'saved_models', 'saved_models/sentiment',
        'backtest_results', 'backtest_results/optimization_results', 'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Python 패키지용 __init__.py 생성
        package_dirs = ['config', 'data', 'models', 'strategy', 'trading', 'sentiment', 'finance', 'utils']
        if directory in package_dirs:
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write(f'# {directory} 패키지\n')
    
    print("✅ 디렉토리 구조 생성 완료")

def create_requirements():
    """requirements.txt 생성"""
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
    print("✅ requirements.txt 생성 완료")

def create_env():
    """.env 파일 생성"""
    env_content = """# 업비트 API 키 설정
UPBIT_ACCESS_KEY=hf0nxaYZb5OVV1bxO3bpQskVsNENft939azTagYu
UPBIT_SECRET_KEY=SitgryqdwGm2xraQZfefeut5Sxjjs49g64Akuunc

# 거래 설정
TRADING_MODE=paper
LOG_LEVEL=INFO

# 리스크 관리
STOP_LOSS_THRESHOLD=0.05
TAKE_PROFIT_THRESHOLD=0.1"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env 파일 생성 완료")
        print("⚠️  .env 파일에 실제 API 키를 입력하세요!")

def create_settings():
    """config/settings.py 생성"""
    settings_content = '''import os
from dotenv import load_dotenv
from datetime import datetime

# 환경 변수 로드
load_dotenv()

# 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
BACKTEST_DIR = os.path.join(BASE_DIR, 'backtest_results')

# 기본 설정
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')

# API 키 (환경 변수에서 로드)
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')

# 리스크 관리
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
    
    with open('config/settings.py', 'w') as f:
        f.write(settings_content)
    print("✅ config/settings.py 생성 완료")

def install_packages():
    """패키지 설치"""
    try:
        print("📦 패키지 설치 중...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ 패키지 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False

def test_setup():
    """설정 테스트"""
    try:
        print("🔍 설정 테스트 중...")
        import pyupbit
        tickers = pyupbit.get_tickers(fiat="KRW")
        print(f"✅ 업비트 연결 성공 ({len(tickers)}개 코인)")
        
        # 비트코인 데이터 테스트
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=5)
        print(f"✅ 데이터 수집 성공 (최근 BTC: {df['close'].iloc[-1]:,.0f}원)")
        return True
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def main():
    print("🚀 가상화폐 자동거래 시스템 설정 시작\n")
    
    # 1. 디렉토리 생성
    create_directories()
    
    # 2. 설정 파일들 생성
    create_requirements()
    create_env()
    create_settings()
    
    print("\n" + "="*50)
    print("📋 다음 단계를 진행하세요:")
    print("1. pip install -r requirements.txt")
    print("2. .env 파일에 API 키 입력")
    print("3. python test_connection.py")
    print("="*50)
    
    # 패키지 설치 여부 확인
    install = input("\n지금 패키지를 설치하시겠습니까? (y/n): ").lower()
    if install in ['y', 'yes']:
        if install_packages():
            if test_setup():
                print("\n🎉 설정 완료! 이제 거래를 시작할 수 있습니다.")
                print("python main.py --mode paper --balance 1000000")

if __name__ == "__main__":
    main()

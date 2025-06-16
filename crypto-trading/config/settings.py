import os
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

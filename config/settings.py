import os
import logging
from dotenv import load_dotenv
from datetime import datetime

# 환경 변수 로드
load_dotenv()

class SecurityConfig:
    """보안 설정 관리"""
    
    @staticmethod
    def validate_api_keys():
        """API 키 유효성 검사"""
        access_key = os.getenv('UPBIT_ACCESS_KEY', '')
        secret_key = os.getenv('UPBIT_SECRET_KEY', '')
        
        if not access_key or access_key == 'your_access_key_here':
            raise ValueError("UPBIT_ACCESS_KEY가 설정되지 않았습니다")
        if not secret_key or secret_key == 'your_secret_key_here':
            raise ValueError("UPBIT_SECRET_KEY가 설정되지 않았습니다")
        if len(access_key) < 20 or len(secret_key) < 20:
            raise ValueError("API 키 형식이 올바르지 않습니다")
        return True
    
    @staticmethod
    def is_safe_mode():
        return os.getenv('ENABLE_SAFETY_CHECKS', 'true').lower() == 'true'

# 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
BACKTEST_DIR = os.path.join(BASE_DIR, 'backtest_results')

# 디렉토리 생성
for directory in [LOG_DIR, DATA_CACHE_DIR, MODEL_DIR, BACKTEST_DIR]:
    os.makedirs(directory, exist_ok=True)

# 기본 설정
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')

# API 키 (보안 검증 포함)
try:
    if TRADING_MODE == 'live':
        SecurityConfig.validate_api_keys()
    UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
    UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')
except ValueError as e:
    print(f"⚠️ API 키 오류: {e}")
    TRADING_MODE = 'paper'
    UPBIT_ACCESS_KEY = ''
    UPBIT_SECRET_KEY = ''

# 리스크 관리 설정
STOP_LOSS_THRESHOLD = min(0.2, float(os.getenv('STOP_LOSS_THRESHOLD', '0.05')))
TAKE_PROFIT_THRESHOLD = min(0.5, float(os.getenv('TAKE_PROFIT_THRESHOLD', '0.1')))
TRAILING_STOP_ACTIVATION = float(os.getenv('TRAILING_STOP_ACTIVATION', '0.05'))
TRAILING_STOP_DISTANCE = float(os.getenv('TRAILING_STOP_DISTANCE', '0.03'))
MAX_POSITION_SIZE = min(0.5, float(os.getenv('MAX_POSITION_SIZE', '0.3')))

# 백테스트 설정
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2023-01-01')
BACKTEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
BACKTEST_INITIAL_BALANCE = int(os.getenv('BACKTEST_INITIAL_BALANCE', '20000000'))
BACKTEST_COMMISSION = float(os.getenv('BACKTEST_COMMISSION', '0.0005'))

# 데이터 설정
DEFAULT_TIMEFRAME = os.getenv('DEFAULT_TIMEFRAME', 'minute30')
DATA_CACHE_EXPIRY = int(os.getenv('DATA_CACHE_EXPIRY', '3600'))  # 초 단위 (1시간)

# 모델 설정
MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0.0')
ML_FEATURES = [
    'ma_ratio_5_20', 'ma_ratio_5_60', 'price_to_ma20', 
    'rsi', 'bb_position', 'macd_hist', 'volume_ratio'
]

# 보안 설정
ENABLE_SAFETY_CHECKS = SecurityConfig.is_safe_mode()
API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '10'))

# 텔레그램 설정 (선택사항)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# 거래 설정
MIN_ORDER_AMOUNT = int(os.getenv('MIN_ORDER_AMOUNT', '5000'))  # 최소 주문 금액
MAX_COINS = int(os.getenv('MAX_COINS', '5'))  # 최대 보유 코인 수
TRADE_INTERVAL = int(os.getenv('TRADE_INTERVAL', '60'))  # 거래 확인 주기 (초)

# 추가 디렉토리 설정
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# 로깅 포맷 설정
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 디버그 모드
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# 설정 확인 출력 (디버그 모드에서만)
if DEBUG:
    print("=== 설정 로드 완료 ===")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"TRADING_MODE: {TRADING_MODE}")
    print(f"LOG_LEVEL: {LOG_LEVEL}")
    print(f"DATA_CACHE_EXPIRY: {DATA_CACHE_EXPIRY}")
    print("==================")
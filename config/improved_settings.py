
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 설정
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')

# 거래 설정
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')
INITIAL_BALANCE = int(os.getenv('INITIAL_BALANCE', '20000000'))

# 로깅 설정
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = 'logs'

# 리스크 관리
STOP_LOSS_THRESHOLD = float(os.getenv('STOP_LOSS_THRESHOLD', '0.05'))
TAKE_PROFIT_THRESHOLD = float(os.getenv('TAKE_PROFIT_THRESHOLD', '0.15'))
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.3'))

# 데이터 품질 기준
MIN_DATA_DAYS = int(os.getenv('MIN_DATA_DAYS', '90'))
MIN_VOLUME_KRW = int(os.getenv('MIN_VOLUME_KRW', '10000000000'))  # 100억원

# ML 모델 설정
MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', '200'))
MODEL_RETRAIN_HOURS = int(os.getenv('MODEL_RETRAIN_HOURS', '24'))  # 24시간마다

# 거래 주기 설정 (초)
TRADING_INTERVAL = int(os.getenv('TRADING_INTERVAL', '3600'))  # 1시간
STATUS_CHECK_INTERVAL = int(os.getenv('STATUS_CHECK_INTERVAL', '1800'))  # 30분

# 텔레그램 설정
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# 캐시 설정
CACHE_DIR = 'data_cache'
MODEL_SAVE_DIR = 'saved_models'

# 백테스팅 설정
BACKTEST_DAYS = int(os.getenv('BACKTEST_DAYS', '30'))

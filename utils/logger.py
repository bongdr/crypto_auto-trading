import logging
import os
from datetime import datetime
from config.settings import LOG_LEVEL, LOG_DIR

def setup_logger(name, level=None, log_file=None):
    """로깅 설정"""
    if level is None:
        level = LOG_LEVEL
        
    # 로그 레벨 문자열을 실제 로깅 레벨로 변환
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
    else:
        numeric_level = level
    
    # 로거 생성 또는 가져오기
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있다면 중복 설정 방지
    if logger.handlers:
        return logger
    
    logger.setLevel(numeric_level)
    
    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # 콘솔 출력 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 출력 핸들러 (옵션)
    if log_file is None:
        # 로그 디렉토리 확인
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 기본 로그 파일명 설정
        current_date = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(LOG_DIR, f"{name}_{current_date}.log")
        
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
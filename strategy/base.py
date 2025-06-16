from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("base_strategy")

class BaseStrategy(ABC):
    """전략 기본 인터페이스"""
    
    def __init__(self, name="Base Strategy"):
        """초기화"""
        self.name = name
        logger.debug(f"{name} 전략 초기화")
    
    @abstractmethod
    def generate_signal(self, df):
        """매매 신호 생성 (자식 클래스에서 구현)
        
        Args:
            df: OHLCV 데이터를 포함한 DataFrame
            
        Returns:
            매매 신호: 1 (매수), -1 (매도), 0 (홀딩)을 포함한 pandas Series
        """
        pass
    
    def get_name(self):
        """전략 이름 반환"""
        return self.name
    
    def __str__(self):
        """문자열 표현"""
        return f"Strategy: {self.name}"

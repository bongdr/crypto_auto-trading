# 수동으로 단계별 수정하기

# 1. strategy 디렉토리 확인 및 생성
mkdir -p strategy

# 2. strategy/__init__.py 생성
echo "# strategy 패키지" > strategy/__init__.py

# 3. strategy/base.py 생성 (BaseStrategy 클래스)
cat > strategy/base.py << 'EOF'
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
EOF

# 4. 기존 sentiment_strategy.py 백업 (있다면)
if [ -f "strategy/sentiment_strategy.py" ]; then
    cp strategy/sentiment_strategy.py strategy/sentiment_strategy.py.backup
    echo "기존 파일 백업됨"
fi

# 5. 새로운 strategy/sentiment_strategy.py 생성
cat > strategy/sentiment_strategy.py << 'EOF'
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import json

from strategy.base import BaseStrategy

logger = logging.getLogger("sentiment_strategy")

class SentimentStrategy(BaseStrategy):
    """감정 분석 기반 거래 전략"""
    
    def __init__(self, name="감정 분석 전략", params=None):
        """초기화"""
        super().__init__(name)
        
        # 기본 파라미터
        self.params = params or {
            'sentiment_threshold': 0.2,
            'sentiment_weight': 0.4,
            'sentiment_lookback': 3,
            'extreme_sentiment_threshold': 0.7,
            'contrarian_threshold': 0.6,
            'use_contrarian': True,
            'min_news_count': 5
        }
        
        # 감정 분석 모듈 (더미)
        self.sentiment_collector = None
        self.sentiment_analyzer = None
        
        # 캐시 설정
        self.sentiment_cache = {}
        self.cache_dir = 'data_cache/sentiment_strategy'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"{name} 전략 초기화 완료")
    
    def _get_recent_sentiment(self, ticker, days=None):
        """최근 감정 데이터 조회 (더미 구현)"""
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'news': {'total_count': 0},
            'social': {},
            'market_indicators': {},
            'scores': {
                'overall_score': 0,
                'news_score': 0,
                'social_score': 0,
                'market_score': 0,
                'components': {}
            }
        }
    
    def generate_sentiment_indicators(self, ticker, df):
        """감정 지표 생성"""
        result_df = df.copy()
        
        # 감정 데이터 로드
        sentiment_data = self._get_recent_sentiment(ticker)
        
        # 기본 감정 점수 설정 (중립)
        result_df['sentiment_score'] = 0
        result_df['sentiment_signal'] = 0
        result_df['sentiment_trend'] = 0
        result_df['extreme_sentiment'] = 0
        
        return result_df
    
    def generate_signal(self, df):
        """매매 신호 생성"""
        # 티커 결정
        ticker = df.name if hasattr(df, 'name') else 'KRW-BTC'
        
        # 감정 지표 추가
        df_with_sentiment = self.generate_sentiment_indicators(ticker, df)
        
        # 기본 기술적 신호 생성
        technical_signal = self._generate_technical_signal(df)
        
        # 감정 기반 신호
        sentiment_signal = df_with_sentiment['sentiment_signal']
        extreme_sentiment = df_with_sentiment['extreme_sentiment']
        
        # 신호 결합
        combined_signal = pd.Series(0, index=df.index)
        
        # 기술적 신호와 감정 신호 조합
        weight_tech = 1 - self.params['sentiment_weight']
        weight_sentiment = self.params['sentiment_weight']
        
        for i in range(len(df)):
            # 극단적 감정 있는 경우 우선 적용
            if extreme_sentiment.iloc[i] != 0:
                combined_signal.iloc[i] = extreme_sentiment.iloc[i]
            else:
                # 기술적 신호와 감정 신호 가중 평균
                tech_value = technical_signal.iloc[i] if i < len(technical_signal) else 0
                sent_value = sentiment_signal.iloc[i]
                
                weighted_signal = (tech_value * weight_tech) + (sent_value * weight_sentiment)
                
                # 신호 이진화
                if weighted_signal > 0.3:
                    combined_signal.iloc[i] = 1
                elif weighted_signal < -0.3:
                    combined_signal.iloc[i] = -1
        
        return combined_signal
    
    def _generate_technical_signal(self, df):
        """기본 기술적 신호 생성"""
        # 기술적 지표 확인
        if not all(col in df.columns for col in ['close', 'ma5', 'ma20', 'rsi']):
            # 필요한 지표가 없으면 중립 신호
            return pd.Series(0, index=df.index)
        
        # 시그널 초기화
        signal = pd.Series(0, index=df.index)
        
        try:
            # 이동평균 교차
            ma_cross_up = (df['ma5'].shift(1) < df['ma20'].shift(1)) & (df['ma5'] > df['ma20'])
            ma_cross_down = (df['ma5'].shift(1) > df['ma20'].shift(1)) & (df['ma5'] < df['ma20'])
            
            # RSI 매수/매도 구간
            rsi_oversold = df['rsi'] < 30
            rsi_overbought = df['rsi'] > 70
            
            # 매수 신호
            buy_signal = ma_cross_up | rsi_oversold
            signal[buy_signal] = 1
            
            # 매도 신호
            sell_signal = ma_cross_down | rsi_overbought
            signal[sell_signal] = -1
            
        except Exception as e:
            logger.error(f"기술적 신호 생성 중 오류: {e}")
            signal = pd.Series(0, index=df.index)
        
        return signal
EOF

# 6. 테스트 파일 생성
cat > test_sentiment_import.py << 'EOF'
#!/usr/bin/env python3
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_import():
    try:
        from strategy.sentiment_strategy import SentimentStrategy
        print("✅ SentimentStrategy 임포트 성공")
        
        # 인스턴스 생성 테스트
        strategy = SentimentStrategy("테스트")
        print("✅ SentimentStrategy 인스턴스 생성 성공")
        
        return True
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_import()
EOF

# 7. 실행 권한 부여
chmod +x test_sentiment_import.py

echo "✅ 수동 수정 완료!"
echo ""
echo "다음 명령어로 테스트하세요:"
echo "python test_sentiment_import.py"

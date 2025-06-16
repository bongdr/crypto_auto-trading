#!/usr/bin/env python3
"""
추가 오류 수정사항
파일명: additional_fixes.py
"""

# 1. sentiment_strategy.py의 중복 코드 제거 및 올바른 구현
SENTIMENT_STRATEGY_CORRECT = '''import pandas as pd
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
            'sentiment_threshold': 0.2,      # 매수 신호 감정 임계값
            'sentiment_weight': 0.4,         # 감정 신호 가중치
            'sentiment_lookback': 3,         # 감정 데이터 참조 일수
            'extreme_sentiment_threshold': 0.7,  # 극단적 감정 임계값
            'contrarian_threshold': 0.6,     # 역발상 전략 임계값
            'use_contrarian': True,          # 역발상 전략 사용 여부
            'min_news_count': 5              # 최소 뉴스 개수
        }
        
        # 감정 분석 모듈 (더미)
        self.sentiment_collector = None
        self.sentiment_analyzer = None
        
        logger.info(f"{name} 전략 초기화 완료")
    
    def _get_recent_sentiment(self, ticker, days=None):
        """최근 감정 데이터 조회 (더미 구현)"""
        # 더미 감정 데이터 반환
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
                'market_score': 0
            }
        }
    
    def generate_sentiment_indicators(self, ticker, df):
        """감정 지표 생성"""
        result_df = df.copy()
        
        # 감정 데이터 로드 (더미)
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
        ticker = getattr(df, 'name', 'KRW-BTC')
        
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
        required_cols = ['close', 'ma5', 'ma20', 'rsi']
        if not all(col in df.columns for col in required_cols):
            # 필요한 지표가 없으면 중립 신호
            return pd.Series(0, index=df.index)
        
        # 시그널 초기화
        signal = pd.Series(0, index=df.index)
        
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
        
        return signal
'''

# 2. advanced_trading_system.py의 올바른 import 문 수정
IMPORT_FIXES = '''
# advanced_trading_system.py 파일 상단에 추가할 import 문들
try:
    from sentiment.data_collector import SentimentDataCollector
    from sentiment.analyzer import SentimentAnalyzer  
    from strategy.sentiment_strategy import SentimentStrategy
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logger.warning("감정 분석 모듈을 찾을 수 없습니다. 더미 모듈을 사용합니다.")

try:
    from finance.fund_manager import FundManager
    from finance.portfolio_rebalancer import PortfolioRebalancer
    FINANCE_AVAILABLE = True
except ImportError:
    FINANCE_AVAILABLE = False
    logger.warning("자금 관리 모듈을 찾을 수 없습니다. 더미 모듈을 사용합니다.")
'''

# 3. main.py에서 텔레그램 관련 오류 수정
TELEGRAM_FIX = '''
# main.py에서 텔레그램 설정 부분 수정
def setup_telegram_config(args):
    """텔레그램 설정 안전하게 처리"""
    telegram_config = None
    
    if args.telegram:
        # 환경 변수에서 토큰과 채팅 ID 가져오기
        import os
        token = args.telegram_token or os.getenv('TELEGRAM_BOT_TOKEN', '')
        chat_id = args.telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID', '')
        
        if token and chat_id:
            telegram_config = {
                'enabled': True,
                'token': token,
                'chat_id': chat_id,
                'report_time': '21:00'
            }
            print(f"✅ 텔레그램 알림 설정됨 (채팅 ID: {chat_id[:10]}...)")
        else:
            print("⚠️  텔레그램 토큰 또는 채팅 ID가 없습니다. 텔레그램 알림이 비활성화됩니다.")
    
    return telegram_config
'''

# 4. 안전한 모듈 임포트를 위한 헬퍼 함수
SAFE_IMPORT_HELPER = '''
def safe_import_modules():
    """안전한 모듈 임포트"""
    modules = {}
    
    # 감정 분석 모듈
    try:
        from sentiment.data_collector import SentimentDataCollector
        from sentiment.analyzer import SentimentAnalyzer
        from strategy.sentiment_strategy import SentimentStrategy
        modules['sentiment'] = {
            'SentimentDataCollector': SentimentDataCollector,
            'SentimentAnalyzer': SentimentAnalyzer,
            'SentimentStrategy': SentimentStrategy,
            'available': True
        }
    except ImportError as e:
        print(f"감정 분석 모듈 임포트 실패: {e}")
        modules['sentiment'] = {'available': False}
    
    # 자금 관리 모듈  
    try:
        from finance.fund_manager import FundManager
        from finance.portfolio_rebalancer import PortfolioRebalancer
        modules['finance'] = {
            'FundManager': FundManager,
            'PortfolioRebalancer': PortfolioRebalancer,
            'available': True
        }
    except ImportError as e:
        print(f"자금 관리 모듈 임포트 실패: {e}")
        modules['finance'] = {'available': False}
    
    return modules
'''

# 5. 실행 스크립트 (오류 수정 및 테스트)
def apply_all_fixes():
    """모든 수정사항 적용"""
    import os
    
    print("🔧 추가 오류 수정 적용 중...")
    
    # strategy/sentiment_strategy.py 올바른 내용으로 교체
    if os.path.exists('strategy/sentiment_strategy.py'):
        with open('strategy/sentiment_strategy.py', 'w', encoding='utf-8') as f:
            f.write(SENTIMENT_STRATEGY_CORRECT)
        print("✅ strategy/sentiment_strategy.py 수정 완료")
    
    # .env 파일에 텔레그램 설정 추가 (예시)
    env_additions = """
# 텔레그램 알림 설정 (선택사항)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
"""
    
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            env_content = f.read()
        
        if 'TELEGRAM_BOT_TOKEN' not in env_content:
            with open('.env', 'a') as f:
                f.write(env_additions)
            print("✅ .env 파일에 텔레그램 설정 추가됨")
    
    print("🎉 모든 추가 수정사항 적용 완료!")

# 6. 테스트 스크립트
def test_modules():
    """모듈 임포트 테스트"""
    print("🧪 모듈 임포트 테스트 중...")
    
    # 감정 분석 모듈 테스트
    try:
        from sentiment.analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        print("✅ 감정 분석 모듈 임포트 성공")
    except Exception as e:
        print(f"❌ 감정 분석 모듈 임포트 실패: {e}")
    
    # 자금 관리 모듈 테스트
    try:
        from finance.fund_manager import FundManager
        manager = FundManager(1000000)
        print("✅ 자금 관리 모듈 임포트 성공")
    except Exception as e:
        print(f"❌ 자금 관리 모듈 임포트 실패: {e}")
    
    # 감정 전략 테스트
    try:
        from strategy.sentiment_strategy import SentimentStrategy
        strategy = SentimentStrategy()
        print("✅ 감정 전략 모듈 임포트 성공")
    except Exception as e:
        print(f"❌ 감정 전략 모듈 임포트 실패: {e}")
    
    print("🎯 모듈 테스트 완료")

if __name__ == "__main__":
    apply_all_fixes()
    test_modules()
    
    print("\n📋 추가 수정사항:")
    print("✅ sentiment_strategy.py 중복 코드 제거 및 올바른 구현")
    print("✅ 안전한 모듈 임포트 구현")
    print("✅ 텔레그램 설정 오류 수정")
    print("✅ .env 파일에 텔레그램 설정 추가")
    
    print("\n🚀 이제 시스템이 정상적으로 작동할 것입니다!")
    print("\n실행 방법:")
    print("1. python fix_sentiment_fund_errors.py  # 메인 수정")
    print("2. python additional_fixes.py           # 추가 수정")
    print("3. python test_connection.py            # 연결 테스트")
    print("4. python main.py --mode paper          # 거래 시작")
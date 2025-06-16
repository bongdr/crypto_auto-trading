import pandas as pd
import numpy as np
from strategy.base import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("technical_strategy")

class TechnicalStrategy(BaseStrategy):
    """기술적 지표 기반 매매 전략"""
    
    def __init__(self, name="기술적 전략", params=None):
        """초기화"""
        super().__init__(name)
        
        # 기본 매개변수
        self.params = params or {
            # 이동평균선 파라미터
            'ma_short': 5,
            'ma_mid': 20,
            'ma_long': 60,
            
            # RSI 파라미터
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # 볼린저 밴드 파라미터 
            'bb_period': 20,
            'bb_std': 2,
            
            # MACD 파라미터
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # 신호 임계값
            'buy_threshold': 0.3,
            'sell_threshold': -0.3,
            
            # 신호 가중치
            'weight_ma_cross': 1.0,
            'weight_rsi': 1.0,
            'weight_bb': 1.0,
            'weight_macd': 1.0,
            'weight_volume': 0.5,
            'weight_stoch': 0.8
        }
    def generate_signal(self, df):
        """매매 신호 생성"""
        # 데이터 유효성 검사
        if df is None or len(df) < max(self.params['ma_long'], 60):
            logger.warning("데이터가 충분하지 않습니다")
            return pd.Series(0, index=df.index)
            
        # 필수 컬럼 확인 (볼린저 밴드 컬럼명 수정)
        required_columns = ['ma5', 'ma20', 'ma60', 'rsi', 'bb_upper', 'bb_lower', 
                           'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'volume_ratio']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"필요한 칼럼이 없습니다: {missing_columns}")
            return pd.Series(0, index=df.index)
        
        # 매매 신호 초기화 (0: 홀딩, 양수: 매수 점수, 음수: 매도 점수)
        signals = pd.Series(0, index=df.index)
        
        # ==== 매수 신호 계산 ====
        
        # 1. 이동평균선 기반 신호
        ma_score = pd.Series(0, index=df.index)
        # 골든 크로스 (단기>중기)
        ma_cross_short_mid = (df['ma5'] > df['ma20'])
        # 중기>장기 확인
        ma_cross_mid_long = (df['ma20'] > df['ma60'])
        # 상승 추세 확인
        ma_uptrend = ma_cross_short_mid & ma_cross_mid_long
        # 단기 이동평균 기울기 확인
        ma_slope = df['ma5'].pct_change(5)
        ma_score[ma_uptrend & (ma_slope > 0)] = 1  # 상승 추세 & 단기 상승
        ma_score[~ma_uptrend & (ma_slope < 0)] = -1  # 하락 추세 & 단기 하락
        
        # 2. RSI 기반 신호
        rsi_score = pd.Series(0, index=df.index)
        # 과매도 구간에서 회복 시 매수 신호
        rsi_oversold_recovery = (df['rsi'] < self.params['rsi_oversold']) & (df['rsi'] > df['rsi'].shift())
        # 과매수 구간에서 하락 시 매도 신호
        rsi_overbought_decline = (df['rsi'] > self.params['rsi_overbought']) & (df['rsi'] < df['rsi'].shift())
        rsi_score[rsi_oversold_recovery] = 1
        rsi_score[rsi_overbought_decline] = -1
        
        # 3. 볼린저 밴드 기반 신호
        bb_score = pd.Series(0, index=df.index)
        # 하단 바운스 (매수 신호)
        bb_bounce = (df['close'].shift() <= df['bb_lower'].shift()) & (df['close'] > df['bb_lower'])
        # 상단 돌파 (매도 신호)
        bb_breakout = (df['close'] >= df['bb_upper'])
        # 중심선 회귀 (극단 벗어날 때)
        bb_position = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        bb_regression = (bb_position < 0.2) | (bb_position > 0.8)
        bb_score[bb_bounce] = 1
        bb_score[bb_breakout] = -1
        bb_score[bb_regression & (bb_position < 0.5)] = 0.5  # 하단에서 중앙으로 회귀 중
        bb_score[bb_regression & (bb_position > 0.5)] = -0.5  # 상단에서 중앙으로 회귀 중
        # 4. MACD 기반 신호
        macd_score = pd.Series(0, index=df.index)
        # MACD 라인이 시그널 라인을 상향 돌파 (매수 신호)
        macd_cross_up = (df['macd'].shift() <= df['macd_signal'].shift()) & (df['macd'] > df['macd_signal'])
        # MACD 라인이 시그널 라인을 하향 돌파 (매도 신호)
        macd_cross_down = (df['macd'].shift() >= df['macd_signal'].shift()) & (df['macd'] < df['macd_signal'])
        # MACD 히스토그램 모멘텀
        macd_hist = df['macd'] - df['macd_signal']
        macd_hist_momentum = macd_hist - macd_hist.shift(3)
        macd_score[macd_cross_up] = 1
        macd_score[macd_cross_down] = -1
        macd_score[macd_hist_momentum > 0] += 0.3  # 모멘텀 상승
        macd_score[macd_hist_momentum < 0] -= 0.3  # 모멘텀 하락
        
        # 5. 스토캐스틱 기반 신호
        stoch_score = pd.Series(0, index=df.index)
        # 과매도 구간에서 %K가 %D를 상향 돌파 (매수 신호)
        stoch_oversold_cross_up = (df['stoch_k'] < 30) & (df['stoch_k'].shift() <= df['stoch_d'].shift()) & (df['stoch_k'] > df['stoch_d'])
        # 과매수 구간에서 %K가 %D를 하향 돌파 (매도 신호)
        stoch_overbought_cross_down = (df['stoch_k'] > 70) & (df['stoch_k'].shift() >= df['stoch_d'].shift()) & (df['stoch_k'] < df['stoch_d'])
        stoch_score[stoch_oversold_cross_up] = 1
        stoch_score[stoch_overbought_cross_down] = -1
        
        # 6. 거래량 기반 신호
        volume_score = pd.Series(0, index=df.index)
        # 거래량 급증 (평균 대비 2배 이상)
        volume_surge = df['volume_ratio'] > 2.0
        # 가격 상승과 함께 거래량 증가
        price_up_volume_up = (df['close'] > df['close'].shift()) & (df['volume'] > df['volume'].shift())
        # 가격 하락과 함께 거래량 증가
        price_down_volume_up = (df['close'] < df['close'].shift()) & (df['volume'] > df['volume'].shift())
        volume_score[volume_surge & price_up_volume_up] = 0.5
        volume_score[volume_surge & price_down_volume_up] = -0.5
        
        # 7. 모든 신호 가중 결합
        signals = (
            ma_score * self.params['weight_ma_cross'] +
            rsi_score * self.params['weight_rsi'] +
            bb_score * self.params['weight_bb'] +
            macd_score * self.params['weight_macd'] +
            stoch_score * self.params['weight_stoch'] +
            volume_score * self.params['weight_volume']
        )
        
        # 8. 신호 이진화 (-1, 0, 1)
        buy_signals = signals > self.params['buy_threshold']
        sell_signals = signals < self.params['sell_threshold']
        neutral = ~(buy_signals | sell_signals)
        
        final_signals = pd.Series(0, index=df.index)
        final_signals[buy_signals] = 1
        final_signals[sell_signals] = -1
        final_signals[neutral] = 0
        
        return final_signals
    
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("feature_engineering")

class FeatureEngineer:
    """기술적 지표 생성 및 특성 공학 수행"""
    
    def __init__(self):
        """초기화"""
        pass
    
    def add_indicators(self, df):
        """OHLCV 데이터에 기술적 지표 추가"""
        if df is None or len(df) < 60:  # 최소 60개 데이터 필요
            logger.warning("데이터가 충분하지 않습니다. 최소 60개 이상 필요.")
            return df
        
        result = df.copy()
        
        try:
            # 이동평균선 (Simple Moving Average)
            result['ma5'] = result['close'].rolling(window=5).mean()
            result['ma10'] = result['close'].rolling(window=10).mean()
            result['ma20'] = result['close'].rolling(window=20).mean()
            result['ma60'] = result['close'].rolling(window=60).mean()
            result['ma120'] = result['close'].rolling(window=120).mean()
            
            # 지수 이동평균선 (Exponential Moving Average)
            result['ema5'] = result['close'].ewm(span=5, adjust=False).mean()
            result['ema10'] = result['close'].ewm(span=10, adjust=False).mean()
            result['ema20'] = result['close'].ewm(span=20, adjust=False).mean()
            
            # 이동평균 수렴/발산 (MACD)
            result['ema12'] = result['close'].ewm(span=12, adjust=False).mean()
            result['ema26'] = result['close'].ewm(span=26, adjust=False).mean()
            result['macd'] = result['ema12'] - result['ema26']
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
            
            # 상대강도지수 (RSI)
            delta = result['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            result['rsi'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드 (Bollinger Bands)
            result['bb_middle'] = result['close'].rolling(window=20).mean()
            result['bb_std'] = result['close'].rolling(window=20).std()
            result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * 2)
            result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * 2)
            result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
            
            # 스토캐스틱 오실레이터 (Stochastic Oscillator)
            high_14 = result['high'].rolling(window=14).max()
            low_14 = result['low'].rolling(window=14).min()
            result['stoch_k'] = ((result['close'] - low_14) / (high_14 - low_14)) * 100
            result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()
            
            # 가격 변화율 (Price Rate of Change)
            result['price_roc_5'] = result['close'].pct_change(periods=5) * 100
            result['price_roc_10'] = result['close'].pct_change(periods=10) * 100
            result['price_roc_20'] = result['close'].pct_change(periods=20) * 100
            
            # Average True Range (ATR)
            tr1 = result['high'] - result['low']
            tr2 = abs(result['high'] - result['close'].shift())
            tr3 = abs(result['low'] - result['close'].shift())
            result['tr'] = pd.DataFrame([tr1, tr2, tr3]).max()
            result['atr'] = result['tr'].rolling(window=14).mean()
            
            # On-Balance Volume (OBV)
            result['obv'] = (np.sign(result['close'].diff()) * result['volume']).fillna(0).cumsum()
            
            # 거래량 이동평균
            result['volume_ma5'] = result['volume'].rolling(window=5).mean()
            result['volume_ma20'] = result['volume'].rolling(window=20).mean()
            result['volume_ratio'] = result['volume'] / result['volume_ma20']
            
            # 가격대 거래량 (Price Volume Trend)
            result['pvt'] = (result['close'].pct_change() * result['volume']).fillna(0).cumsum()
            
            # 추세 강도 지표 (ADX - Average Directional Index)
            # 양의 방향 이동 (+DM)
            result['plus_dm'] = result['high'].diff()
            result['plus_dm'] = result['plus_dm'].where(
                (result['plus_dm'] > 0) & (result['plus_dm'] > -result['low'].diff()), 0
            )
            
            # 음의 방향 이동 (-DM)
            result['minus_dm'] = -result['low'].diff()
            result['minus_dm'] = result['minus_dm'].where(
                (result['minus_dm'] > 0) & (result['minus_dm'] > result['high'].diff()), 0
            )
            
            # TR 기반 +DI, -DI 계산
            result['plus_di'] = 100 * (result['plus_dm'].rolling(window=14).mean() / result['atr'])
            result['minus_di'] = 100 * (result['minus_dm'].rolling(window=14).mean() / result['atr'])
            
            # ADX 계산
            result['dx'] = 100 * abs(result['plus_di'] - result['minus_di']) / (result['plus_di'] + result['minus_di'])
            result['adx'] = result['dx'].rolling(window=14).mean()
            
            # 추세 방향 지표
            result['trend_strength'] = abs(result['ma5'] / result['ma20'] - 1)
            result['trend_direction'] = np.where(result['ma5'] > result['ma20'], 1, -1)
            
            # 캔들스틱 패턴 지표
            result['body_size'] = abs(result['open'] - result['close'])
            result['shadow_ratio'] = (result['high'] - result['low']) / (result['body_size'] + 0.001)
            result['upper_shadow'] = result['high'] - result[['open', 'close']].max(axis=1)
            result['lower_shadow'] = result[['open', 'close']].min(axis=1) - result['low']
            
            # 가격 모멘텀
            result['momentum'] = result['close'] - result['close'].shift(10)
            
            # ROC (Rate of Change)
            result['roc'] = result['close'].pct_change(periods=10) * 100
            
            # Williams %R
            highest_high = result['high'].rolling(window=14).max()
            lowest_low = result['low'].rolling(window=14).min()
            result['williams_r'] = ((highest_high - result['close']) / (highest_high - lowest_low)) * -100
            
            # Commodity Channel Index (CCI)
            typical_price = (result['high'] + result['low'] + result['close']) / 3
            mean_dev = abs(typical_price - typical_price.rolling(window=20).mean()).rolling(window=20).mean()
            result['cci'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_dev)
            
            # 경향선 지표 (추세의 각도)
            result['close_lag5'] = result['close'].shift(5)
            result['angle'] = np.arctan((result['close'] / result['close_lag5'] - 1) * 100) * (180 / np.pi)
            
            # 레이블 인코딩 및 무한값 처리
            result = result.replace([np.inf, -np.inf], np.nan)
            result = result.fillna(method='ffill').fillna(method='bfill')
            
            logger.debug(f"기술적 지표 {len(result.columns)}개 추가됨")
            return result
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 오류: {e}")
            return df
    
    def add_ml_features(self, df):
        """머신러닝 모델을 위한 추가 특성 생성"""
        if df is None or len(df) < 60:
            logger.warning("ML 특성 생성을 위한 데이터가 부족합니다")
            return None
            
        try:
            # 기본 기술적 지표 추가
            result = self.add_indicators(df)
            
            # 특성 상호작용 및 변환
            result['ma_ratio_5_20'] = result['ma5'] / result['ma20']
            result['ma_ratio_20_60'] = result['ma20'] / result['ma60']
            
            result['rsi_ma'] = result['rsi'].rolling(window=5).mean()
            result['rsi_slope'] = result['rsi'] - result['rsi'].shift(3)
            
            # 볼린저 밴드 위치
            result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
            
            result['volume_ma_ratio'] = result['volume_ma5'] / result['volume_ma20']
            
            result['candle_pattern'] = ((result['close'] - result['open']) / 
                                      (result['high'] - result['low'] + 0.001)) * 100
            
            # 다양한 시간대 변화율
            for period in [1, 3, 5, 10, 20]:
                result[f'return_{period}d'] = result['close'].pct_change(periods=period)
                result[f'volume_change_{period}d'] = result['volume'].pct_change(periods=period)
            
            # 기술적 지표 교차 특성
            result['ma_cross'] = np.where(
                (result['ma5'].shift(1) < result['ma20'].shift(1)) & 
                (result['ma5'] > result['ma20']), 
                1, 
                np.where(
                    (result['ma5'].shift(1) > result['ma20'].shift(1)) & 
                    (result['ma5'] < result['ma20']), 
                    -1, 
                    0
                )
            )
            
            # 스토캐스틱 K와 D의 교차
            result['stoch_cross'] = np.where(
                (result['stoch_k'].shift(1) < result['stoch_d'].shift(1)) & 
                (result['stoch_k'] > result['stoch_d']), 
                1, 
                np.where(
                    (result['stoch_k'].shift(1) > result['stoch_d'].shift(1)) & 
                    (result['stoch_k'] < result['stoch_d']), 
                    -1, 
                    0
                )
            )
            
            # MACD 히스토그램 방향 전환
            result['macd_cross'] = np.where(
                (result['macd_hist'].shift(1) < 0) & (result['macd_hist'] > 0), 
                1, 
                np.where(
                    (result['macd_hist'].shift(1) > 0) & (result['macd_hist'] < 0), 
                    -1, 
                    0
                )
            )
            
            # 볼린저 밴드 돌파
            result['bb_breakout_up'] = np.where(result['close'] > result['bb_upper'], 1, 0)
            result['bb_breakout_down'] = np.where(result['close'] < result['bb_lower'], 1, 0)
            
            # 추가 변환 및 스케일링
            result['close_to_max_20d'] = result['close'] / result['high'].rolling(window=20).max()
            result['close_to_min_20d'] = result['close'] / result['low'].rolling(window=20).min()
            
            # 무한값 및 결측치 처리
            result = result.replace([np.inf, -np.inf], np.nan)
            result = result.fillna(method='ffill').fillna(method='bfill')
            
            logger.debug(f"ML 특성 {len(result.columns)}개 생성됨")
            return result
            
        except Exception as e:
            logger.error(f"ML 특성 생성 오류: {e}")
            return df
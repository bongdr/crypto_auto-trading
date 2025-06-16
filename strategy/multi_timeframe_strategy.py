import pandas as pd
import numpy as np
from strategy.base import BaseStrategy
from strategy.technical import TechnicalStrategy
from utils.logger import setup_logger

logger = setup_logger("multi_timeframe_strategy")

class MultiTimeframeStrategy(BaseStrategy):
    """여러 시간대 데이터를 활용하는 전략"""
    
    def __init__(self, name="멀티 타임프레임 전략", tf_weights=None):
        """초기화"""
        super().__init__(name)
        
        # 각 시간프레임 별 가중치 (기본값: 일봉 > 시간봉 > 분봉)
        self.tf_weights = tf_weights or {
            'day': 3.0,    # 일봉
            'hour4': 2.0,  # 4시간봉
            'hour': 1.5,   # 1시간봉
            'minute30': 1.0  # 30분봉
        }
        
        # 각 시간프레임 별 전략
        self.strategies = {}
        for tf in self.tf_weights.keys():
            self.strategies[tf] = TechnicalStrategy(f"{tf} 기술적 전략")
    
    def set_data(self, data_dict):
        """각 시간프레임 별 데이터 설정"""
        self.data = data_dict
    
    def add_strategy(self, timeframe, strategy):
        """특정 시간프레임에 커스텀 전략 추가"""
        if timeframe not in self.tf_weights:
            logger.warning(f"지원하지 않는 시간프레임: {timeframe}")
            return self
        
        self.strategies[timeframe] = strategy
        logger.info(f"{timeframe} 시간프레임에 {strategy.get_name()} 전략 추가됨")
        return self
    
    def set_timeframe_weight(self, timeframe, weight):
        """시간프레임 가중치 설정"""
        if timeframe not in self.tf_weights:
            logger.warning(f"지원하지 않는 시간프레임: {timeframe}")
            return
        
        self.tf_weights[timeframe] = weight
        logger.info(f"{timeframe} 시간프레임 가중치 {weight}로 설정됨")
    
    def align_timeframes(self, base_df, signals_dict):
        """다른 시간프레임의 신호를 기준 데이터프레임 인덱스에 맞춤"""
        # 기준 인덱스 (일반적으로 가장 짧은 시간프레임)
        base_index = base_df.index
        
        aligned_signals = {}
        for tf, signals in signals_dict.items():
            # 새로운 시리즈 생성 (기본값 0)
            aligned = pd.Series(0, index=base_index)
            
            # 각 기준 인덱스에 대해 해당 시간에 유효한 상위 시간프레임 신호 찾기
            for idx in base_index:
                # 상위 시간프레임에서 현재 시간 이전의 가장 최근 신호 찾기
                valid_indices = signals.index[signals.index <= idx]
                if len(valid_indices) > 0:
                    latest_idx = valid_indices[-1]
                    aligned[idx] = signals[latest_idx]
            
            aligned_signals[tf] = aligned
        
        return aligned_signals
    
    def generate_signal(self, df):
        """여러 시간프레임의 신호를 결합하여 최종 신호 생성"""
        # 단일 데이터프레임만 제공된 경우 (기본 전략 사용)
        if not hasattr(self, 'data') or not self.data:
            logger.warning("멀티 타임프레임 데이터가 설정되지 않았습니다")
            # 기본 시간프레임(일봉)의 전략만 사용
            if 'day' in self.strategies:
                return self.strategies['day'].generate_signal(df)
            else:
                # 기본 기술적 전략으로 대체
                default_strategy = TechnicalStrategy("기본 전략")
                return default_strategy.generate_signal(df)
            
            # 각 시간프레임 별 신호 생성
        signals_dict = {}
        for tf, tf_data in self.data.items():
            if tf in self.strategies:
                signals = self.strategies[tf].generate_signal(tf_data)
                signals_dict[tf] = signals
                logger.debug(f"{tf} 시간프레임 신호 생성: {sum(signals != 0)} 개")
            else:
                logger.warning(f"{tf} 시간프레임에 대한 전략이 없습니다")
        
        # 신호가 없으면 기본값 반환
        if not signals_dict:
            return pd.Series(0, index=df.index)
        
        # 기준 시간프레임에 다른 시간프레임 신호 정렬
        aligned_signals = self.align_timeframes(df, signals_dict)
        
        # 가중 평균 계산
        weighted_sum = pd.Series(0, index=df.index)
        total_weight = 0
        
        for tf, signals in aligned_signals.items():
            if tf in self.tf_weights:
                weight = self.tf_weights[tf]
                weighted_sum += signals * weight
                total_weight += weight
        
        # 정규화
        if total_weight > 0:
            weighted_sum /= total_weight
        
        # 최종 신호 생성
        final_signals = pd.Series(0, index=df.index)
        final_signals[weighted_sum > 0.3] = 1  # 매수 임계값
        final_signals[weighted_sum < -0.3] = -1  # 매도 임계값
        
        # 신호 통계
        buy_count = sum(final_signals == 1)
        sell_count = sum(final_signals == -1)
        hold_count = len(final_signals) - buy_count - sell_count
        
        logger.debug(f"멀티 타임프레임 최종 신호: 매수 {buy_count}, 매도 {sell_count}, 홀딩 {hold_count}")
        
        return final_signals
    
    def trend_confirmation_strategy(self, df):
        """트렌드 확인 전략 (상위 시간프레임 트렌드 방향을 따르는 하위 시간프레임 신호만 승인)"""
        if not hasattr(self, 'data') or not self.data:
            logger.warning("멀티 타임프레임 데이터가 설정되지 않았습니다")
            return pd.Series(0, index=df.index)
        
        # 시간프레임 정렬 (큰 것부터)
        timeframes = sorted(self.data.keys(), 
                          key=lambda x: (0 if x == 'day' else 
                                       (1 if x.startswith('hour') else 2)), 
                          reverse=True)
        
        # 각 시간프레임 별 트렌드 방향 계산
        trend_directions = {}
        for tf in timeframes:
            tf_data = self.data[tf]
            
            # 이동평균선 기반 트렌드 방향
            if 'ma20' not in tf_data.columns or 'ma60' not in tf_data.columns:
                # 필요한 지표 추가
                from models.feature_engineering import FeatureEngineer
                fe = FeatureEngineer()
                tf_data = fe.add_indicators(tf_data)
            
            # 1: 상승 트렌드, -1: 하락 트렌드, 0: 횡보
            trend = np.where(tf_data['ma20'] > tf_data['ma60'], 1, 
                            np.where(tf_data['ma20'] < tf_data['ma60'], -1, 0))
            
            trend_directions[tf] = pd.Series(trend, index=tf_data.index)
        
        # 상위 시간프레임 트렌드를 기준 인덱스에 맞춤
        aligned_trends = self.align_timeframes(df, trend_directions)
        
        # 가장 짧은 시간프레임의 매매 신호 생성
        shortest_tf = timeframes[-1]
        if shortest_tf in self.strategies:
            base_signals = self.strategies[shortest_tf].generate_signal(self.data[shortest_tf])
        else:
            # 기본 전략으로 대체
            base_strategy = TechnicalStrategy("기본 전략")
            base_signals = base_strategy.generate_signal(self.data[shortest_tf])
        
        # 기준 인덱스에 맞춤
        base_signals_aligned = pd.Series(0, index=df.index)
        common_idx = df.index.intersection(base_signals.index)
        base_signals_aligned.loc[common_idx] = base_signals.loc[common_idx]
        
        # 상위 시간프레임 트렌드 방향과 일치하는 신호만 승인
        final_signals = pd.Series(0, index=df.index)
        
        # 가장 상위 시간프레임 (일반적으로 일봉)
        highest_tf = timeframes[0]
        highest_trend = aligned_trends[highest_tf]
        
        # 매수 신호: 상위 시간프레임이 상승 추세일 때만 승인
        buy_signal = (base_signals_aligned == 1) & (highest_trend > 0)
        
        # 매도 신호: 상위 시간프레임이 하락 추세일 때만 승인
        sell_signal = (base_signals_aligned == -1) & (highest_trend < 0)
        
        final_signals[buy_signal] = 1
        final_signals[sell_signal] = -1
        
        return final_signals
    
    def pullback_strategy(self, df):
        """풀백 전략 (상위 시간프레임 추세 방향의 하위 시간프레임 조정 시 진입)"""
        if not hasattr(self, 'data') or not self.data:
            logger.warning("멀티 타임프레임 데이터가 설정되지 않았습니다")
            return pd.Series(0, index=df.index)
        
        # 시간프레임 정렬 (큰 것부터)
        timeframes = sorted(self.data.keys(), 
                          key=lambda x: (0 if x == 'day' else 
                                       (1 if x.startswith('hour') else 2)), 
                          reverse=True)
        
        if len(timeframes) < 2:
            logger.warning("풀백 전략을 위해서는 최소 2개 이상의 시간프레임이 필요합니다")
            return pd.Series(0, index=df.index)
        
        # 상위 시간프레임 추세 방향 계산
        higher_tf = timeframes[0]
        higher_data = self.data[higher_tf]
        
        if 'ma20' not in higher_data.columns or 'ma60' not in higher_data.columns:
            # 필요한 지표 추가
            from models.feature_engineering import FeatureEngineer
            fe = FeatureEngineer()
            higher_data = fe.add_indicators(higher_data)
        
        # 상위 시간프레임 추세 (1: 상승, -1: 하락)
        higher_trend = np.where(higher_data['ma20'] > higher_data['ma60'], 1, 
                              np.where(higher_data['ma20'] < higher_data['ma60'], -1, 0))
        higher_trend = pd.Series(higher_trend, index=higher_data.index)
        
        # 하위 시간프레임 RSI 계산
        lower_tf = timeframes[-1]
        lower_data = self.data[lower_tf]
        
        if 'rsi' not in lower_data.columns:
            # 필요한 지표 추가
            from models.feature_engineering import FeatureEngineer
            fe = FeatureEngineer()
            lower_data = fe.add_indicators(lower_data)
        
        # 기준 인덱스에 맞춤
        aligned_higher_trend = pd.Series(0, index=df.index)
        for idx in df.index:
            # 현재 시간 이전의 가장 최근 상위 시간프레임 추세 찾기
            valid_indices = higher_trend.index[higher_trend.index <= idx]
            if len(valid_indices) > 0:
                latest_idx = valid_indices[-1]
                aligned_higher_trend[idx] = higher_trend[latest_idx]
        
        aligned_lower_rsi = pd.Series(0, index=df.index)
        common_idx = df.index.intersection(lower_data.index)
        aligned_lower_rsi.loc[common_idx] = lower_data.loc[common_idx, 'rsi']
        
        # 풀백 조건 (상승 추세 내 과매도, 하락 추세 내 과매수)
        final_signals = pd.Series(0, index=df.index)
        
        # 상승 추세 내 과매도 (RSI < 40) 시 매수
        buy_condition = (aligned_higher_trend > 0) & (aligned_lower_rsi < 40)
        
        # 하락 추세 내 과매수 (RSI > 60) 시 매도
        sell_condition = (aligned_higher_trend < 0) & (aligned_lower_rsi > 60)
        
        final_signals[buy_condition] = 1
        final_signals[sell_condition] = -1
        
        return final_signals
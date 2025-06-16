import pandas as pd
import numpy as np
from strategy.base import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger("ensemble_strategy")

class EnsembleStrategy(BaseStrategy):
    """여러 전략을 결합하는 앙상블 전략"""
    
    def __init__(self, name="앙상블 전략"):
        super().__init__(name)
        self.strategies = []  # (전략, 가중치) 튜플 리스트
    
    def add_strategy(self, strategy, weight=1.0):
        """전략 추가"""
        self.strategies.append((strategy, weight))
        logger.info(f"전략 추가됨: {strategy.get_name()}, 가중치: {weight}")
        return self
    
    def generate_signal(self, df):
        """모든 전략의 가중 평균 신호 생성"""
        if not self.strategies:
            logger.warning("등록된 전략이 없습니다")
            return pd.Series(0, index=df.index)
        
        # 각 전략에서 신호 생성
        all_signals = []
        total_weight = 0
        
        for strategy, weight in self.strategies:
            signal = strategy.generate_signal(df)
            
            # 인덱스 맞추기
            common_index = signal.index.intersection(df.index)
            aligned_signal = pd.Series(0, index=df.index)
            aligned_signal.loc[common_index] = signal.loc[common_index]
            
            all_signals.append((aligned_signal, weight))
            total_weight += weight
            
            logger.debug(f"{strategy.get_name()} 신호 생성 완료: {sum(signal != 0)} 신호")
        
        # 가중 평균 계산
        weighted_sum = pd.Series(0, index=df.index)
        for signal, weight in all_signals:
            weighted_sum += signal * weight
        
        # 가중치 정규화
        if total_weight > 0:
            weighted_sum /= total_weight
            
        # 최종 신호 생성 (임계값 기반)
        buy_threshold = 0.3
        sell_threshold = -0.3
        
        final_signals = pd.Series(0, index=df.index)
        final_signals[weighted_sum > buy_threshold] = 1
        final_signals[weighted_sum < sell_threshold] = -1
        
        # 신호 요약
        buy_count = sum(final_signals == 1)
        sell_count = sum(final_signals == -1)
        logger.debug(f"앙상블 신호: 매수 {buy_count}, 매도 {sell_count}, 유지 {len(final_signals) - buy_count - sell_count}")
        
        return final_signals
    
    def vote_based_signal(self, df, buy_threshold=0.5, sell_threshold=0.5):
        """투표 기반 앙상블 (다수결)"""
        if not self.strategies:
            logger.warning("등록된 전략이 없습니다")
            return pd.Series(0, index=df.index)
        
        # 각 전략에서 신호 생성
        buy_votes = pd.Series(0, index=df.index)
        sell_votes = pd.Series(0, index=df.index)
        
        for strategy, weight in self.strategies:
            signal = strategy.generate_signal(df)
            
            # 인덱스 맞추기
            common_index = signal.index.intersection(df.index)
            
            # 매수/매도 투표 집계
            buy_votes.loc[common_index] += (signal.loc[common_index] == 1) * weight
            sell_votes.loc[common_index] += (signal.loc[common_index] == -1) * weight
        
        # 총 가중치 계산
        total_weight = sum(weight for _, weight in self.strategies)
        
        # 투표 비율 계산
        buy_ratio = buy_votes / total_weight
        sell_ratio = sell_votes / total_weight
        
        # 임계값 기반 최종 결정
        final_signals = pd.Series(0, index=df.index)
        final_signals[buy_ratio >= buy_threshold] = 1
        final_signals[sell_ratio >= sell_threshold] = -1
        
        # 충돌 해결 (매수/매도 동시에 투표된 경우)
        # 투표율이 더 높은 쪽 선택
        conflict = (final_signals == 1) & (final_signals == -1)
        final_signals[conflict & (buy_ratio > sell_ratio)] = 1
        final_signals[conflict & (buy_ratio <= sell_ratio)] = -1
        
        return final_signals
    
    def stacked_ensemble(self, df, meta_strategy=None):
        """스택된 앙상블 (메타 전략 사용)"""
        if not self.strategies:
            logger.warning("등록된 전략이 없습니다")
            return pd.Series(0, index=df.index)
        
        # 각 전략에서 신호 생성하여 특성으로 사용
        strategy_signals = pd.DataFrame(index=df.index)
        
        for i, (strategy, _) in enumerate(self.strategies):
            signal = strategy.generate_signal(df)
            
            # 인덱스 맞추기
            common_index = signal.index.intersection(df.index)
            
            # 기본 신호
            strategy_signals[f"strategy_{i}_signal"] = 0
            strategy_signals.loc[common_index, f"strategy_{i}_signal"] = signal.loc[common_index]
            
            # 추가 특성: 이전 신호들
            for j in range(1, 4):
                strategy_signals[f"strategy_{i}_signal_lag{j}"] = strategy_signals[f"strategy_{i}_signal"].shift(j)
        
        # NaN 처리
        strategy_signals = strategy_signals.fillna(0)
        
        # 메타 전략이 없으면 단순 가중 평균 사용
        if meta_strategy is None:
            # 단순 신호 평균
            signal_cols = [col for col in strategy_signals.columns if col.endswith('_signal')]
            avg_signal = strategy_signals[signal_cols].mean(axis=1)
            
            final_signals = pd.Series(0, index=df.index)
            final_signals[avg_signal > 0.3] = 1
            final_signals[avg_signal < -0.3] = -1
            
        else:
            # 메타 전략 사용하여 최종 신호 결정
            final_signals = meta_strategy.generate_signal(strategy_signals)
        
        return final_signals
    
    def dynamic_weight_ensemble(self, df, lookback=30):
        """성과 기반 동적 가중치 앙상블"""
        if not self.strategies:
            logger.warning("등록된 전략이 없습니다")
            return pd.Series(0, index=df.index)
        
        # 각 전략에서 신호 생성
        strategy_signals = []
        for strategy, _ in self.strategies:
            signal = strategy.generate_signal(df)
            strategy_signals.append(signal)
        
        # 인덱스 및 길이 맞추기
        common_index = df.index
        for signal in strategy_signals:
            common_index = common_index.intersection(signal.index)
        
        # 빈 인덱스면 기본값 반환
        if len(common_index) == 0:
            return pd.Series(0, index=df.index)
        
        # 가중치 계산 위한 성과 측정
        weights = []
        aligned_signals = []
        
        for i, signal in enumerate(strategy_signals):
            aligned_signal = pd.Series(0, index=df.index)
            aligned_signal.loc[common_index] = signal.loc[common_index]
            aligned_signals.append(aligned_signal)
            
            # 과거 성과 평가
            if len(common_index) > lookback:
                performance_index = common_index[-lookback:]
                
                # 신호에 따른 수익 계산
                returns = df.loc[performance_index, 'close'].pct_change()
                signal_returns = aligned_signal.loc[performance_index].shift(1) * returns
                
                # 성과 지표 계산 (샤프 비율)
                if len(signal_returns.dropna()) > 0:
                    avg_return = signal_returns.mean()
                    std_return = signal_returns.std()
                    sharpe = avg_return / std_return if std_return > 0 else 0
                    
                    # 음수는 0으로, 나머지는 지수함수로 가중치 증폭
                    weight = np.exp(max(0, sharpe))
                else:
                    weight = 1.0
            else:
                weight = 1.0
                
            weights.append(weight)
        
        # 가중치 정규화
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        
        # 가중 평균 계산
        weighted_sum = pd.Series(0, index=df.index)
        for signal, weight in zip(aligned_signals, normalized_weights):
            weighted_sum += signal * weight
        
        # 최종 신호 생성
        final_signals = pd.Series(0, index=df.index)
        final_signals[weighted_sum > 0.3] = 1
        final_signals[weighted_sum < -0.3] = -1
        
        # 성과 로깅
        for i, (strategy, _) in enumerate(self.strategies):
            logger.debug(f"{strategy.get_name()}: 가중치 {normalized_weights[i]:.4f}")
        
        return final_signals
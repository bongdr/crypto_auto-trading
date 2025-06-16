import numpy as np
import pandas as pd
from collections import deque
import joblib
import os
from datetime import datetime
from threading import Thread
import time

from strategy.base import BaseStrategy
from utils.logger import setup_logger
from config.settings import MODEL_DIR

logger = setup_logger("adaptive_ensemble")

class AdaptiveEnsemble(BaseStrategy):
    """전략 가중치 자동 조정 앙상블 전략"""
    
    def __init__(self, name="적응형 앙상블", learning_rate=0.05, history_window=100):
        """초기화"""
        super().__init__(name)
        self.strategies = []  # (전략, 가중치, 성과 히스토리) 튜플 리스트
        self.learning_rate = learning_rate  # 가중치 업데이트 속도
        self.history_window = history_window  # 성과 추적 윈도우 크기
        
        # 가중치 업데이트 모드
        self.update_mode = 'online'  # 'online', 'bandit', 'reinforce' 중 선택
        
        # 최근 신호 및 실제 수익률 저장 (온라인 학습용)
        self.recent_signals = {}  # {timestamp: {ticker: signal}} 
        self.actual_returns = {}  # {ticker: {timestamp: return}}
        
        # 멀티암드 밴딧 알고리즘 변수 (UCB 방식)
        self.strategy_rewards = []  # 각 전략의 보상 기록
        self.strategy_pulls = []    # 각 전략의 선택 횟수
        self.exploration_weight = 2.0  # 탐색 가중치 (높을수록 탐색 강화)
        
        # 저장/로드 설정
        self.save_path = os.path.join(MODEL_DIR, "adaptive_ensemble.joblib")
        
        # 자동 업데이트 설정
        self.running = False
        self.update_interval = 3600  # 초 단위 (1시간)
    
    def add_strategy(self, strategy, initial_weight=1.0):
        """전략 추가"""
        # 성과 히스토리 초기화 (최근 N개 거래의 수익률)
        history = deque(maxlen=self.history_window)
        
        self.strategies.append((strategy, initial_weight, history))
        
        # 밴딧 알고리즘 변수 초기화
        if self.update_mode == 'bandit':
            self.strategy_rewards.append([])
            self.strategy_pulls.append(0)
            
        logger.info(f"전략 추가됨: {strategy.get_name()}, 초기 가중치: {initial_weight}")
        return self
    
    def generate_signal(self, df):
        """모든 전략의 가중 평균 신호 생성"""
        if not self.strategies:
            logger.warning("등록된 전략이 없습니다")
            return pd.Series(0, index=df.index)
        
        # 각 전략에서 신호 생성
        all_signals = []
        total_weight = 0
        
        for i, (strategy, weight, _) in enumerate(self.strategies):
            signal = strategy.generate_signal(df)
            
            # 인덱스 맞추기
            common_index = signal.index.intersection(df.index)
            aligned_signal = pd.Series(0, index=df.index)
            aligned_signal.loc[common_index] = signal.loc[common_index]
            
            # 밴딧 알고리즘일 경우 가중치 계산
            if self.update_mode == 'bandit':
                # UCB(Upper Confidence Bound) 가중치 계산
                weight = self._calculate_ucb_weight(i)
                
            all_signals.append((aligned_signal, weight))
            total_weight += weight
            
            # 최근 신호 저장 (온라인 학습용)
            if len(common_index) > 0:
                latest_idx = common_index[-1]
                latest_time = latest_idx.to_pydatetime()
                
                if latest_time not in self.recent_signals:
                    self.recent_signals[latest_time] = {}
                
                ticker = df.name if hasattr(df, 'name') else 'unknown'
                self.recent_signals[latest_time][ticker] = (aligned_signal.iloc[-1], i)  # 신호와 전략 인덱스 저장

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
    
    def _calculate_ucb_weight(self, strategy_idx):
        """UCB 알고리즘으로 전략 가중치 계산"""
        # 전략별 평균 보상
        rewards = self.strategy_rewards[strategy_idx]
        pulls = self.strategy_pulls[strategy_idx]
        
        if pulls == 0:  # 한 번도 선택되지 않은 전략
            return 10.0  # 높은 초기 가중치로 탐색 유도
            
        # 평균 보상
        avg_reward = sum(rewards) / pulls if rewards else 0
        
        # 총 시행 횟수
        total_pulls = sum(self.strategy_pulls)
        
        # UCB 가중치: 평균 보상 + 탐색 항
        if total_pulls > 0:
            exploration_term = self.exploration_weight * np.sqrt(np.log(total_pulls) / pulls)
            ucb_weight = avg_reward + exploration_term
            
            # 음수 가중치 방지
            ucb_weight = max(0.1, ucb_weight)
            
            return ucb_weight
        
        return 1.0  # 기본 가중치
    
    def update_weights_online(self, ticker, timestamp, actual_return):
        """온라인 학습 방식으로 가중치 업데이트"""
        if timestamp not in self.recent_signals or ticker not in self.recent_signals[timestamp]:
            return
        
        signal, strategy_idx = self.recent_signals[timestamp][ticker]
        
        # 신호가 0이면 업데이트 불필요
        if signal == 0:
            return
            
        # 각 전략의 성과 히스토리 업데이트
        strategy, weight, history = self.strategies[strategy_idx]
        
        # 신호의 정확성 확인
        signal_correct = (signal == 1 and actual_return > 0) or (signal == -1 and actual_return < 0)
        
        # 성과 히스토리 업데이트
        history.append(1 if signal_correct else -1)
        self.strategies[strategy_idx] = (strategy, weight, history)
        
        # 최근 성과에 기반한 가중치 업데이트
        if len(history) >= 5:  # 최소 5개 이상의 데이터가 있어야 함
            recent_performance = sum(list(history)[-5:]) / 5  # 최근 5개 성과의 평균
            
            # 새 가중치 계산
            new_weight = weight * (1 + self.learning_rate * recent_performance)
            
            # 음수 가중치 방지
            new_weight = max(0.1, new_weight)
            
            # 가중치 업데이트
            self.strategies[strategy_idx] = (strategy, new_weight, history)
            
            logger.debug(f"{strategy.get_name()} 가중치 업데이트: {weight:.4f} → {new_weight:.4f}")
        
        # 밴딧 알고리즘 통계 업데이트
        if self.update_mode == 'bandit':
            self.strategy_rewards[strategy_idx].append(actual_return)
            self.strategy_pulls[strategy_idx] += 1    
            
    def record_actual_return(self, ticker, timestamp, price, next_price):
        """실제 수익률 기록"""
        # 수익률 계산
        actual_return = (next_price / price) - 1
        
        # 저장
        if ticker not in self.actual_returns:
            self.actual_returns[ticker] = {}
            
        self.actual_returns[ticker][timestamp] = actual_return
        
        # 가중치 업데이트
        self.update_weights_online(ticker, timestamp, actual_return)
        
        # 오래된 데이터 정리 (메모리 관리)
        self._clean_old_data()
        
        return actual_return
    
    def _clean_old_data(self):
        """오래된 데이터 정리"""
        # 신호 데이터 정리 (최대 100개만 유지)
        if len(self.recent_signals) > 100:
            old_keys = sorted(self.recent_signals.keys())[:len(self.recent_signals) - 100]
            for key in old_keys:
                self.recent_signals.pop(key, None)
        
        # 수익률 데이터 정리
        for ticker in self.actual_returns:
            if len(self.actual_returns[ticker]) > 100:
                ticker_timestamps = sorted(self.actual_returns[ticker].keys())
                old_keys = ticker_timestamps[:len(ticker_timestamps) - 100]
                for key in old_keys:
                    self.actual_returns[ticker].pop(key, None)
    
    def save_state(self):
        """현재 가중치 상태 저장"""
        try:
            # 저장할 데이터 준비
            save_data = {
                'weights': [weight for _, weight, _ in self.strategies],
                'strategy_names': [strategy.get_name() for strategy, _, _ in self.strategies],
                'update_mode': self.update_mode,
                'learning_rate': self.learning_rate,
                'saved_at': datetime.now(),
                'bandit_data': {
                    'rewards': self.strategy_rewards,
                    'pulls': self.strategy_pulls
                } if self.update_mode == 'bandit' else None
            }
            
            # 디렉토리 확인
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # 저장
            joblib.dump(save_data, self.save_path)
            logger.info(f"앙상블 상태 저장 완료: {self.save_path}")
            return True
            
        except Exception as e:
            logger.error(f"앙상블 상태 저장 실패: {e}")
            return False
    
    def load_state(self, strategies_list):
        """저장된 가중치 상태 로드"""
        if not os.path.exists(self.save_path):
            logger.warning(f"저장된 상태 파일 없음: {self.save_path}")
            return False
            
        try:
            # 로드
            save_data = joblib.load(self.save_path)
            
            # 기존 전략과 매핑
            if len(save_data['strategy_names']) != len(strategies_list):
                logger.warning(f"전략 수 불일치: 저장됨 {len(save_data['strategy_names'])}, 현재 {len(strategies_list)}")
                return False
                
            # 가중치 복원
            for i, (strategy, _, history) in enumerate(self.strategies):
                expected_name = save_data['strategy_names'][i]
                if strategy.get_name() == expected_name:
                    self.strategies[i] = (strategy, save_data['weights'][i], history)
                else:
                    logger.warning(f"전략 이름 불일치: 기대 {expected_name}, 실제 {strategy.get_name()}")
            
            # 밴딧 알고리즘 데이터 복원
            if 'bandit_data' in save_data and save_data['bandit_data']:
                self.strategy_rewards = save_data['bandit_data']['rewards']
                self.strategy_pulls = save_data['bandit_data']['pulls']
            
            # 학습률 복원
            self.learning_rate = save_data.get('learning_rate', self.learning_rate)
            self.update_mode = save_data.get('update_mode', self.update_mode)
            
            logger.info(f"앙상블 상태 로드 완료: {self.save_path}, 마지막 저장: {save_data.get('saved_at')}")
            return True
            
        except Exception as e:
            logger.error(f"앙상블 상태 로드 실패: {e}")
            return False
        
    def start_auto_update(self):
        """자동 업데이트 시작"""
        if self.running:
            return
            
        self.running = True
        
        # 백그라운드 스레드 시작
        update_thread = Thread(target=self._update_loop)
        update_thread.daemon = True
        update_thread.start()
        
        logger.info("앙상블 자동 업데이트 시작됨")
    
    def _update_loop(self):
        """가중치 자동 업데이트 루프"""
        while self.running:
            try:
                # 가중치 정규화 (모든 가중치의 합이 전략 수와 같도록)
                total_weight = sum(weight for _, weight, _ in self.strategies)
                if total_weight > 0:
                    norm_factor = len(self.strategies) / total_weight
                    
                    for i, (strategy, weight, history) in enumerate(self.strategies):
                        normalized_weight = weight * norm_factor
                        self.strategies[i] = (strategy, normalized_weight, history)
                    
                    logger.debug("앙상블 가중치 정규화 완료")
                
                # 현재 상태 저장
                self.save_state()
                
                # 일정 시간 대기
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"앙상블 업데이트 오류: {e}")
                time.sleep(60)  # 오류 발생 시 1분 후 재시도
    
    def stop_auto_update(self):
        """자동 업데이트 중지"""
        self.running = False
        logger.info("앙상블 자동 업데이트 중지됨")
    
    def get_strategy_weights(self):
        """현재 전략별 가중치 반환"""
        return [(strategy.get_name(), weight) for strategy, weight, _ in self.strategies]
    
    def set_update_mode(self, mode):
        """업데이트 모드 설정"""
        valid_modes = ['online', 'bandit', 'reinforce']
        if mode not in valid_modes:
            logger.warning(f"유효하지 않은 업데이트 모드: {mode}, {valid_modes} 중에서 선택하세요")
            return False
            
        # 모드 변경
        self.update_mode = mode
        logger.info(f"앙상블 업데이트 모드 변경: {mode}")
        
        # 밴딧 모드로 변경 시 초기화
        if mode == 'bandit' and (not self.strategy_rewards or len(self.strategy_rewards) != len(self.strategies)):
            self.strategy_rewards = [[] for _ in self.strategies]
            self.strategy_pulls = [0 for _ in self.strategies]
            
        return True
    
    def set_learning_rate(self, rate):
        """학습률 설정"""
        if rate <= 0 or rate > 1.0:
            logger.warning(f"유효하지 않은 학습률: {rate}, 0 < rate <= 1.0 이어야 함")
            return False
            
        self.learning_rate = rate
        logger.info(f"앙상블 학습률 변경: {rate}")
        return True

# 사용 예시
if __name__ == "__main__":
    from strategy.technical import TechnicalStrategy
    
    # 전략 생성
    tech1 = TechnicalStrategy("이동평균 전략", {'weight_ma_cross': 1.5, 'weight_rsi': 0.5})
    tech2 = TechnicalStrategy("RSI 전략", {'weight_ma_cross': 0.5, 'weight_rsi': 1.5})
    tech3 = TechnicalStrategy("볼린저 전략", {'weight_ma_cross': 0.5, 'weight_bb': 1.5})
    
    # 적응형 앙상블 생성
    adaptive = AdaptiveEnsemble(learning_rate=0.1)
    adaptive.add_strategy(tech1, 1.0)
    adaptive.add_strategy(tech2, 1.0)
    adaptive.add_strategy(tech3, 1.0)
    
    # 온라인 학습 모드 설정
    adaptive.set_update_mode('online')
    
    # 자동 업데이트 시작
    adaptive.start_auto_update()
    
    print("적응형 앙상블이 백그라운드에서 실행 중입니다. Ctrl+C로 중지하세요.")
    try:
        while True:
            time.sleep(10)
            # 현재 가중치 출력
            for name, weight in adaptive.get_strategy_weights():
                print(f"{name}: {weight:.4f}")
            print("------------------------")
    except KeyboardInterrupt:
        adaptive.stop_auto_update()
        print("프로그램이 종료되었습니다.")
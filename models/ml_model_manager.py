# 2. models/ml_model_manager.py 수정 - 무한 재시도 방지
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import joblib
import time
import schedule
from threading import Thread

from data.collector import UpbitDataCollector
from models.feature_engineering import FeatureEngineer
from strategy.ml_strategy import MLStrategy
from utils.logger import setup_logger
from config.settings import MODEL_DIR

logger = setup_logger("ml_model_manager")

class MLModelManager:
    """머신러닝 모델 관리 및 자동 재학습 - 개선된 버전"""
    
    def __init__(self, tickers=None, base_model_path=None):
        """초기화"""
        self.tickers = tickers or ["KRW-BTC"]
        self.base_model_path = base_model_path or MODEL_DIR
        self.collector = UpbitDataCollector()
        self.fe = FeatureEngineer()
        self.models = {}  # {ticker: {'model': MLStrategy객체, 'last_trained': datetime, 'performance': float}}
        self.running = False
        self.performance_threshold = 0.55  # 모델 성능 임계값 (정확도)
        self.market_change_threshold = 0.3  # 시장 변화 감지 임계값
        self.last_market_state = {}  # {ticker: {'volatility': float, 'volume': float}}
        
        # 실패 추적 추가
        self.failed_attempts = {}  # {ticker: {'count': int, 'last_attempt': datetime}}
        self.max_daily_attempts = 3  # 일일 최대 재시도 횟수
        self.min_retry_interval = 3600  # 최소 재시도 간격 (1시간)
        
        # 모델 로드 또는 초기화
        self.initialize_models()
        
    def _can_attempt_training(self, ticker):
        """재학습 시도 가능 여부 확인"""
        if ticker not in self.failed_attempts:
            return True
            
        attempts = self.failed_attempts[ticker]
        now = datetime.now()
        
        # 오늘 날짜 확인
        if attempts['last_attempt'].date() != now.date():
            # 새로운 날이면 카운트 리셋
            self.failed_attempts[ticker] = {'count': 0, 'last_attempt': now}
            return True
        
        # 일일 최대 시도 횟수 확인
        if attempts['count'] >= self.max_daily_attempts:
            logger.info(f"{ticker} 일일 최대 재시도 횟수 초과 ({attempts['count']}/{self.max_daily_attempts})")
            return False
        
        # 최소 재시도 간격 확인
        time_since_last = (now - attempts['last_attempt']).total_seconds()
        if time_since_last < self.min_retry_interval:
            logger.debug(f"{ticker} 재시도 간격 부족: {time_since_last/60:.1f}분 < {self.min_retry_interval/60}분")
            return False
        
        return True
    
    def _record_training_attempt(self, ticker, success=False):
        """학습 시도 기록"""
        now = datetime.now()
        
        if ticker not in self.failed_attempts:
            self.failed_attempts[ticker] = {'count': 0, 'last_attempt': now}
        
        if success:
            # 성공하면 카운트 리셋
            self.failed_attempts[ticker] = {'count': 0, 'last_attempt': now}
        else:
            # 실패하면 카운트 증가
            if self.failed_attempts[ticker]['last_attempt'].date() == now.date():
                self.failed_attempts[ticker]['count'] += 1
            else:
                self.failed_attempts[ticker]['count'] = 1
            self.failed_attempts[ticker]['last_attempt'] = now
        
    def initialize_models(self):
        """모델 초기화"""
        for ticker in self.tickers:
            model_path = os.path.join(self.base_model_path, f"{ticker}_ml_model.joblib")
            
            # 모델 파일이 있는지 확인
            if os.path.exists(model_path):
                ml_strategy = MLStrategy(f"{ticker} 전략", model_path=model_path)
                last_trained = datetime.fromtimestamp(os.path.getmtime(model_path))
                
                self.models[ticker] = {
                    'model': ml_strategy,
                    'last_trained': last_trained,
                    'performance': getattr(ml_strategy, 'test_accuracy', 0.5)
                }
                logger.info(f"{ticker} 모델 로드 완료, 마지막 학습: {last_trained}")
            else:
                # 모델이 없으면 새로 학습
                logger.info(f"{ticker} 모델이 없습니다. 새로 학습합니다.")
                self.train_model(ticker)
    
    def train_model(self, ticker):
        """모델 학습 - 개선된 버전"""
        # 재시도 가능 여부 확인
        if not self._can_attempt_training(ticker):
            return False
        
        logger.info(f"{ticker} 모델 학습 시작")
        model_path = os.path.join(self.base_model_path, f"{ticker}_ml_model.joblib")
        
        try:
            # 더 많은 데이터 수집 시도
            train_df = None
            
            # 1차 시도: 120일 데이터
            train_df = self.collector.get_historical_data(ticker, days=120, interval="day")
            
            # 2차 시도: 90일 데이터
            if train_df is None or len(train_df) < 100:
                logger.info(f"{ticker} 90일 데이터로 재시도")
                train_df = self.collector.get_historical_data(ticker, days=90, interval="day")
            
            # 3차 시도: 시간봉 데이터
            if train_df is None or len(train_df) < 100:
                logger.info(f"{ticker} 시간봉 데이터로 시도")
                train_df = self.collector.get_historical_data(ticker, days=30, interval="minute240")
            
            # 최종 확인
            if train_df is None or len(train_df) < 80:  # 최소 기준 완화
                logger.error(f"{ticker} 학습 데이터 부족: {len(train_df) if train_df is not None else 0} < 80")
                self._record_training_attempt(ticker, success=False)
                return False
            
            # 특성 추가
            train_df_features = self.fe.add_ml_features(train_df)
            if train_df_features is None or len(train_df_features) < 80:
                logger.error(f"{ticker} 특성 생성 실패")
                self._record_training_attempt(ticker, success=False)
                return False
                
            # 학습 모델 생성
            ml_strategy = MLStrategy(f"{ticker} 전략")
            
            # 모델 학습 - 파라미터 조정
            success = ml_strategy.train_model(
                train_df_features,
                horizon=2,  # 예측 기간 단축
                threshold=0.008,  # 임계값 완화
                test_size=0.15,  # 테스트 비율 축소
                save_path=model_path
            )
            
            if success:
                # 성능 개선 계산
                old_performance = self.models.get(ticker, {}).get('performance', 0)
                new_performance = getattr(ml_strategy, 'test_accuracy', 0.5)
                improvement = new_performance - old_performance
                
                self.models[ticker] = {
                    'model': ml_strategy,
                    'last_trained': datetime.now(),
                    'performance': new_performance
                }
                
                logger.info(f"{ticker} 모델 학습 완료, 성능: {new_performance:.4f} (개선: {improvement:+.4f})")
                self._record_training_attempt(ticker, success=True)
                return True
            else:
                logger.error(f"{ticker} 모델 학습 실패")
                self._record_training_attempt(ticker, success=False)
                return False
                
        except Exception as e:
            logger.error(f"{ticker} 모델 학습 중 예외: {e}")
            self._record_training_attempt(ticker, success=False)
            return False
        
    def evaluate_model(self, ticker):
        """모델 성능 평가"""
        if ticker not in self.models:
            logger.warning(f"{ticker} 모델이 없습니다")
            return 0.5
            
        # 최근 데이터 수집
        test_df = self.collector.get_ohlcv(ticker, interval="day", count=30)
        if test_df is None or len(test_df) < 20:
            logger.warning(f"{ticker} 테스트 데이터 부족")
            return 0.5
            
        # 특성 추가
        test_df_features = self.fe.add_ml_features(test_df)
        if test_df_features is None:
            logger.warning(f"{ticker} 테스트 특성 생성 실패")
            return 0.5
            
        # 과거 데이터에 대한 예측 생성
        ml_strategy = self.models[ticker]['model']
        signals = ml_strategy.generate_signal(test_df_features)
        
        # 실제 수익률 계산 (매수 신호가 실제로 수익을 냈는지)
        accuracy = 0.5
        if len(signals) > 0:
            profit_count = 0
            signal_count = 0
            
            for i in range(len(signals) - 3):  # 3일 후 가격 확인
                if signals.iloc[i] == 1:  # 매수 신호
                    signal_count += 1
                    future_return = test_df_features['close'].iloc[min(i+3, len(test_df_features)-1)] / test_df_features['close'].iloc[i] - 1
                    if future_return > 0.01:  # 1% 이상 상승 시 정확
                        profit_count += 1
            
            if signal_count > 0:
                accuracy = profit_count / signal_count
                
        # 모델 성능 업데이트
        self.models[ticker]['performance'] = accuracy
        logger.info(f"{ticker} 모델 성능 평가: {accuracy:.4f}")
        
        return accuracy
    
    def check_market_change(self, ticker):
        """시장 상황 변화 감지"""
        # 최근 데이터 수집
        recent_df = self.collector.get_ohlcv(ticker, interval="day", count=20)
        if recent_df is None or len(recent_df) < 10:
            return False
            
        # 변동성 계산
        current_volatility = recent_df['close'].pct_change().std()
        
        # 거래량 변화 계산
        current_volume = recent_df['volume'].mean()
        
        # 이전 상태와 비교
        if ticker in self.last_market_state:
            prev_state = self.last_market_state[ticker]
            volatility_change = abs(current_volatility / prev_state['volatility'] - 1)
            volume_change = abs(current_volume / prev_state['volume'] - 1)
            
            # 변화 감지
            if volatility_change > self.market_change_threshold or volume_change > self.market_change_threshold:
                logger.info(f"{ticker} 시장 상황 변화 감지: 변동성 변화 {volatility_change:.2f}, 거래량 변화 {volume_change:.2f}")
                
                # 상태 업데이트
                self.last_market_state[ticker] = {
                    'volatility': current_volatility,
                    'volume': current_volume
                }
                
                return True
        
        # 초기 상태 저장
        self.last_market_state[ticker] = {
            'volatility': current_volatility,
            'volume': current_volume
        }
        
        return False
    
    def check_retraining_needed(self, ticker):
        """재학습 필요성 확인 - 실패 기록 고려"""
        if ticker not in self.models:
            return self._can_attempt_training(ticker)  # 모델이 없고 시도 가능한 경우만
            
        model_info = self.models[ticker]
        
        # 1. 시간 기반: 마지막 학습으로부터 7일 이상 지났는지
        time_based = (datetime.now() - model_info['last_trained']).days >= 7
        
        # 2. 성능 기반: 모델 성능이 임계값 이하인지
        performance = self.evaluate_model(ticker)
        performance_based = performance < self.performance_threshold
        
        # 3. 시장 변화 기반: 시장 상황이 크게 변했는지
        market_based = self.check_market_change(ticker)
        
        # 재학습 필요성 판단
        needs_retraining = (time_based or performance_based or market_based) and self._can_attempt_training(ticker)
        
        if needs_retraining:
            reason = "시간 경과" if time_based else ("성능 저하" if performance_based else "시장 변화")
            logger.info(f"{ticker} 모델 재학습 필요 ({reason})")
        
        return needs_retraining
    
    def update_models(self):
        """모든 모델 확인 및 필요 시 재학습"""
        for ticker in self.tickers:
            if self.check_retraining_needed(ticker):
                self.train_model(ticker)
    
    def schedule_daily_update(self):
        """일일 업데이트 스케줄링"""
        schedule.every().day.at("01:00").do(self.update_models)  # 새벽 1시에 실행
        logger.info("일일 모델 업데이트가 스케줄링되었습니다 (01:00)")
    
    def schedule_weekly_update(self):
        """주간 업데이트 스케줄링"""
        schedule.every().monday.at("02:00").do(self.update_models)  # 월요일 새벽 2시에 실행
        logger.info("주간 모델 업데이트가 스케줄링되었습니다 (월요일 02:00)")
    
    def run_scheduler(self):
        """스케줄러 실행"""
        self.running = True
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)
    
    def start(self):
        """관리자 시작"""
        # 스케줄링 설정
        self.schedule_daily_update()
        
        # 백그라운드 스레드로 스케줄러 실행
        scheduler_thread = Thread(target=self.run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info("ML 모델 관리자 시작됨")
    
    def stop(self):
        """관리자 중지"""
        self.running = False
        logger.info("ML 모델 관리자 중지됨")
    
    def get_model(self, ticker):
        """특정 티커의 모델 반환"""
        if ticker in self.models:
            return self.models[ticker]['model']
        return None

# 사용 예시:
if __name__ == "__main__":
    # 모델 관리자 초기화
    tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
    model_manager = MLModelManager(tickers)
    
    # 관리자 시작
    model_manager.start()
    
    try:
        # 메인 스레드는 다른 작업 수행 가능
        print("ML 모델 관리자가 백그라운드에서 실행 중입니다. Ctrl+C로 중지하세요.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # 관리자 종료
        model_manager.stop()
        print("프로그램이 종료되었습니다.")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
from threading import Thread
import time
import random
from collections import deque
import json

from data.collector import UpbitDataCollector
from models.feature_engineering import FeatureEngineer
from strategy.base import BaseStrategy
from utils.logger import setup_logger
from config.settings import MODEL_DIR

logger = setup_logger("market_state_detector")

class MarketState:
    """시장 상태 정보"""
    
    def __init__(self, state_id, features=None, characteristics=None):
        """초기화"""
        self.state_id = state_id
        self.features = features or {}
        self.characteristics = characteristics or {}
        self.occurrences = 1
        self.last_seen = datetime.now()
        self.optimal_strategy = None
        
    def update(self, features=None, characteristics=None):
        """상태 정보 업데이트"""
        if features:
            self.features = features
            
        if characteristics:
            self.characteristics = characteristics
            
        self.last_seen = datetime.now()
        self.occurrences += 1
        
    def get_state_summary(self):
        """상태 요약 정보"""
        summary = {
            'state_id': self.state_id,
            'characteristics': self.characteristics,
            'occurrences': self.occurrences,
            'last_seen': self.last_seen,
            'optimal_strategy': self.optimal_strategy
        }
        return summary
    
    def __str__(self):
        """문자열 표현"""
        chars = []
        for key, value in self.characteristics.items():
            if isinstance(value, float):
                chars.append(f"{key}: {value:.2f}")
            else:
                chars.append(f"{key}: {value}")
                
        return f"상태 {self.state_id}: {', '.join(chars)}, 발생 {self.occurrences}회"

class MarketStateDetector:
    """시장 상태 감지 모듈"""
    
    def __init__(self, ticker="KRW-BTC", detection_method='clustering', n_states=5):
        """초기화"""
        self.ticker = ticker
        self.detection_method = detection_method
        self.n_states = n_states
        
        # 데이터 수집 및 특성 엔지니어링
        self.collector = UpbitDataCollector()
        self.fe = FeatureEngineer()
        
        # 데이터 수집 주기
        self.data_timeframe = 'day'
        self.data_count = 60
        
        # 모델 및 상태 정보
        self.model = None
        self.scaler = StandardScaler()
        self.market_states = {}
        self.current_state = None
        self.previous_state_id = None
        self.state_history = deque(maxlen=100)
        
        # 상태 평가 지표
        self.state_metrics = {}
        
        # CUSUM 변수
        self.cusum_threshold = 1.0
        self.cusum_pos = 0
        self.cusum_neg = 0
        self.cusum_baseline = None
        
        # 자동 실행
        self.running = False
        self.update_interval = 3600
        
        # 모델 저장 경로
        self.save_dir = os.path.join(MODEL_DIR, "market_states")
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info(f"MarketStateDetector 초기화 완료: {self.ticker}, 방법: {self.detection_method}")
    
    def collect_data(self):
        """시장 데이터 수집"""
        try:
            if self.data_timeframe == 'day':
                df = self.collector.get_ohlcv(self.ticker, interval="day", count=self.data_count)
            elif self.data_timeframe == 'hour':
                df = self.collector.get_ohlcv(self.ticker, interval="minute60", count=self.data_count)
            elif self.data_timeframe == 'minute30':
                df = self.collector.get_ohlcv(self.ticker, interval="minute30", count=self.data_count)
            else:
                logger.error(f"지원하지 않는 시간프레임: {self.data_timeframe}")
                return None
                
            if df is None or len(df) < self.data_count * 0.8:
                logger.error(f"데이터 부족: {len(df) if df is not None else 0} 행")
                return None
                
            # 특성 추가
            df = self.fe.add_indicators(df)
            
            logger.debug(f"데이터 수집 완료: {len(df)} 행")
            return df
            
        except Exception as e:
            logger.error(f"데이터 수집 중 오류: {e}")
            return None
    
    def extract_features(self, df):
        """시장 특성 추출"""
        if df is None or len(df) < 5:
            return None
            
        features = {}
        
        try:
            # 1. 추세 특성
            features['trend_direction'] = 1 if df['close'].iloc[-1] > df['ma20'].iloc[-1] else -1
            features['trend_strength'] = abs(df['close'].iloc[-1] / df['ma20'].iloc[-1] - 1)
            
            # 2. 변동성 특성
            features['volatility'] = df['close'].pct_change().std() * np.sqrt(252)
            features['daily_range'] = ((df['high'] - df['low']) / df['low']).mean() * 100
            
            # 3. 모멘텀 특성
            features['rsi'] = df['rsi'].iloc[-1]
            features['macd_hist'] = df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else 0
            
            # 4. 볼린저 밴드 위치
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_upper = df['bb_upper'].iloc[-1]
                bb_lower = df['bb_lower'].iloc[-1]
                close = df['close'].iloc[-1]
                if bb_upper != bb_lower:
                    features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
                else:
                    features['bb_position'] = 0.5
            else:
                features['bb_position'] = 0.5
            
            # 5. 거래량 특성
            if len(df) >= 10:
                features['volume_ratio'] = df['volume'].iloc[-5:].mean() / df['volume'].iloc[:-5].mean()
            else:
                features['volume_ratio'] = 1.0
            
            return features
            
        except Exception as e:
            logger.error(f"특성 추출 중 오류: {e}")
            return None
    
    def characterize_market_state(self, features):
        """시장 상태 특성화"""
        if features is None:
            return None
            
        characteristics = {}
        
        try:
            # 추세 방향 및 강도
            if features['trend_direction'] > 0:
                if features['trend_strength'] > 0.05:
                    characteristics['trend'] = "강한 상승"
                else:
                    characteristics['trend'] = "약한 상승"
            else:
                if features['trend_strength'] > 0.05:
                    characteristics['trend'] = "강한 하락"
                else:
                    characteristics['trend'] = "약한 하락"
                    
            # 변동성
            if features['volatility'] > 0.8:
                characteristics['volatility'] = "매우 높음"
            elif features['volatility'] > 0.5:
                characteristics['volatility'] = "높음"
            elif features['volatility'] > 0.3:
                characteristics['volatility'] = "보통"
            else:
                characteristics['volatility'] = "낮음"
                
            # 과매수/과매도
            if features['rsi'] > 70:
                characteristics['momentum'] = "과매수"
            elif features['rsi'] < 30:
                characteristics['momentum'] = "과매도"
            else:
                characteristics['momentum'] = "중립"
                
            # 거래량
            if features['volume_ratio'] > 1.5:
                characteristics['volume'] = "급증"
            elif features['volume_ratio'] < 0.7:
                characteristics['volume'] = "급감"
            else:
                characteristics['volume'] = "보통"
                
            return characteristics
            
        except Exception as e:
            logger.error(f"상태 특성화 중 오류: {e}")
            return {}
    
    def detect_state_clustering(self, df):
        """군집화 기반 시장 상태 감지"""
        if df is None or len(df) < 10:
            logger.error("군집화를 위한 데이터 부족")
            return self._create_default_state()
            
        try:
            # 기본 특성 사용
            features_list = []
            for i in range(min(len(df), 50)):  # 최근 50개 데이터만 사용
                row_features = []
                row = df.iloc[-(i+1)]  # 최신부터 역순으로
                
                # 기본 특성들
                row_features.append(row['close'] / row['ma20'] if row['ma20'] > 0 else 1.0)
                row_features.append(row['rsi'] / 100.0)
                row_features.append(row['volume'] / row['volume'] if row['volume'] > 0 else 1.0)
                
                # 변동성
                if i < len(df) - 1:
                    volatility = abs(row['close'] / df.iloc[-(i+2)]['close'] - 1)
                    row_features.append(volatility)
                else:
                    row_features.append(0.01)
                
                features_list.append(row_features)
            
            if len(features_list) < 5:
                logger.warning("특성 데이터 부족")
                return self._create_default_state()
            
            features_array = np.array(features_list)
            
            # 모델 학습 또는 로드
            if self.model is None:
                # 새 모델 학습
                self.scaler = StandardScaler()
                scaled_features = self.scaler.fit_transform(features_array)
                
                self.model = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
                self.model.fit(scaled_features)
                
                logger.info(f"새 군집화 모델 학습 완료: {self.n_states}개 군집")
            
            # 현재 상태 예측
            current_features = np.array([features_list[0]]).reshape(1, -1)
            scaled_current = self.scaler.transform(current_features)
            state_id = int(self.model.predict(scaled_current)[0])
            
            # 특성 추출 및 상태 특성화
            features_dict = self.extract_features(df)
            characteristics = self.characterize_market_state(features_dict)
            
            # 상태 객체 업데이트 또는 생성
            if state_id in self.market_states:
                state = self.market_states[state_id]
                state.update(features_dict, characteristics)
            else:
                state = MarketState(state_id, features_dict, characteristics)
                self.market_states[state_id] = state
                logger.info(f"새 시장 상태 감지: {state}")
            
            # 상태 기록
            self.state_history.append(state_id)
            self.current_state = state
            
            return state
            
        except Exception as e:
            logger.error(f"상태 감지 중 오류: {e}")
            return self._create_default_state()
    
    def _create_default_state(self):
        """기본 시장 상태 생성"""
        state_id = "default"
        features = {
            'trend_direction': 1,
            'trend_strength': 0.01,
            'volatility': 0.5,
            'rsi': 50,
            'volume_ratio': 1.0
        }
        characteristics = {
            "trend": "중립",
            "volatility": "보통", 
            "momentum": "중립",
            "volume": "보통"
        }
        default_state = MarketState(state_id, features, characteristics)
        self.market_states[state_id] = default_state
        self.current_state = default_state
        logger.info("기본 시장 상태 생성됨")
        return default_state
    
    def detect_current_state(self):
        """현재 시장 상태 감지"""
        # 이전 상태 저장
        if self.current_state:
            self.previous_state_id = self.current_state.state_id
        
        # 데이터 수집
        df = self.collect_data()
        
        if df is None or len(df) < 10:
            logger.warning("데이터 부족으로 기본 상태 사용")
            return self._create_default_state()
        
        # 감지 방법에 따라 다른 알고리즘 사용
        try:
            if self.detection_method == 'clustering':
                new_state = self.detect_state_clustering(df)
            else:
                logger.warning(f"지원하지 않는 감지 방법: {self.detection_method}, clustering으로 대체")
                new_state = self.detect_state_clustering(df)
                
        except Exception as e:
            logger.error(f"상태 감지 중 예외 발생: {e}")
            new_state = self._create_default_state()
        
        # 최종 안전장치
        if new_state is None:
            logger.warning("모든 상태 감지 방법 실패, 기본 상태 사용")
            new_state = self._create_default_state()
        
        # 상태가 변경된 경우에만 previous_state_id 업데이트
        if new_state and self.current_state and new_state.state_id != self.current_state.state_id:
            self.previous_state_id = self.current_state.state_id
            logger.info(f"{self.ticker} 상태 변경: {self.previous_state_id} → {new_state.state_id}")
        
        return new_state
    
    def get_optimal_strategy(self, state_id=None):
        """특정 상태에 최적화된 전략 반환"""
        if state_id is None:
            if self.current_state is None:
                return None
            state_id = self.current_state.state_id
            
        if state_id not in self.market_states:
            logger.warning(f"알 수 없는 상태 ID: {state_id}")
            return None
            
        state = self.market_states[state_id]
        
        # 최적 전략이 설정되어 있으면 반환
        if state.optimal_strategy:
            return state.optimal_strategy
            
        # 설정되어 있지 않으면 상태 특성 기반으로 추론
        characteristics = state.characteristics
        
        if not characteristics:
            return None
            
        strategy_info = {}
        
        # 추세 기반 전략 선택
        if 'trend' in characteristics:
            if characteristics['trend'] in ["강한 상승", "약한 상승"]:
                strategy_info['type'] = 'trend_following'
                strategy_info['params'] = {
                    'weight_ma_cross': 1.5,
                    'weight_rsi': 0.7,
                    'buy_threshold': 0.3
                }
            elif characteristics['trend'] in ["강한 하락", "약한 하락"]:
                strategy_info['type'] = 'counter_trend'
                strategy_info['params'] = {
                    'weight_rsi': 1.5,
                    'weight_bb': 1.2,
                    'buy_threshold': 0.7
                }
        
        # 상태 업데이트
        state.optimal_strategy = strategy_info
        
        return strategy_info
    
    def start_auto_detection(self):
        """자동 상태 감지 시작"""
        if self.running:
            return
            
        self.running = True
        
        # 백그라운드 스레드 시작
        detection_thread = Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        logger.info("시장 상태 감지기 시작됨")
    
    def _detection_loop(self):
        """상태 감지 루프"""
        while self.running:
            try:
                # 현재 상태 감지
                current_state = self.detect_current_state()
                
                if current_state:
                    logger.debug(f"현재 시장 상태: {current_state}")
                
                # 일정 시간 대기
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"상태 감지 오류: {e}")
                time.sleep(60)  # 오류 시 1분 후 재시도
    
    def stop_auto_detection(self):
        """자동 상태 감지 중지"""
        self.running = False
        logger.info("시장 상태 감지기 중지됨")
    
    def get_current_state_info(self):
        """현재 상태 정보 반환"""
        if self.current_state is None:
            return None
            
        return self.current_state.get_state_summary()
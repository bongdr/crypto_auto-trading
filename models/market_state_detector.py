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
        
        # 🔧 이 부분을 완전히 제거하거나 주석 처리하세요
        """
        # 감지 방법 검증 및 설정 - 이 부분이 문제!
        valid_methods = ['clustering', 'hmm', 'cusum']
        if detection_method not in valid_methods:  # detection_method가 없음!
            logger.warning(f"유효하지 않은 감지 방법: {detection_method}, clustering으로 설정")
            detection_method = 'clustering'
        """
        
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
        self.n_states = n_states  # 군집 또는 상태 수
        
        # 🔧 감지 방법 검증을 여기서 해야 함 (MarketState가 아니라)
        valid_methods = ['clustering', 'hmm', 'cusum']
        if detection_method not in valid_methods:
            logger.warning(f"유효하지 않은 감지 방법: {detection_method}, clustering으로 설정")
            detection_method = 'clustering'
        
        self.detection_method = detection_method
        
        # 데이터 수집 및 특성 엔지니어링
        self.collector = UpbitDataCollector()
        self.fe = FeatureEngineer()
        
        # 데이터 수집 주기
        self.data_timeframe = 'day'  # 'day', 'hour', 'minute30'
        self.data_count = 60  # 최근 60개 데이터
        
        # 모델 및 상태 정보
        self.model = None  # 군집 또는 HMM 모델
        self.scaler = StandardScaler()  # 특성 스케일링
        self.market_states = {}  # 상태 ID -> MarketState 객체
        self.current_state = None  # 현재 시장 상태
        self.previous_state_id = None  # 이 줄 추가
        self.state_history = deque(maxlen=100)  # 최근 100개 상태 기록
        
        # 상태 평가 지표
        self.state_metrics = {}  # 상태 ID -> {전략 -> 수익률 통계}
        
        # CUSUM 변수
        self.cusum_threshold = 1.0  # 변화 감지 임계값
        self.cusum_pos = 0  # 양의 누적합
        self.cusum_neg = 0  # 음의 누적합
        self.cusum_baseline = None  # 기준값
        
        # 자동 실행
        self.running = False
        self.update_interval = 3600  # 1시간마다 업데이트
        
        # 모델 저장 경로
        self.save_dir = os.path.join(MODEL_DIR, "market_states")
        os.makedirs(self.save_dir, exist_ok=True)
        
    def collect_data(self):
        """시장 데이터 수집"""
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
    
    def extract_features(self, df):
        """시장 특성 추출"""
        if df is None or len(df) < 5:  # 최소 5개 데이터 필요
            return None
            
        features = {}
        
        # 1. 추세 특성
        features['trend_direction'] = 1 if df['close'].iloc[-1] > df['ma20'].iloc[-1] else -1
        features['trend_strength'] = abs(df['close'].iloc[-1] / df['ma20'].iloc[-1] - 1)
        
        # 2. 변동성 특성
        features['volatility'] = df['close'].pct_change().std() * np.sqrt(252)  # 연간화
        features['daily_range'] = ((df['high'] - df['low']) / df['low']).mean() * 100
        
        # 3. 모멘텀 특성
        features['rsi'] = df['rsi'].iloc[-1]
        features['macd_hist'] = df['macd_hist'].iloc[-1]
        
        # 4. 볼린저 밴드 위치
        if 'bb_position' in df.columns:
            features['bb_position'] = df['bb_position'].iloc[-1]
        else:
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            close = df['close'].iloc[-1]
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # 5. 거래량 특성
        features['volume_ratio'] = df['volume'].iloc[-5:].mean() / df['volume'].iloc[:-5].mean()
        
        return features
    
    def characterize_market_state(self, features):
        """시장 상태 특성화"""
        if features is None:
            return None
            
        characteristics = {}
        
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
    
    def detect_state_clustering(self, df):
        """군집화 기반 시장 상태 감지"""
        if df is None or len(df) < 10:
            logger.error("군집화를 위한 데이터 부족")
            return self._create_default_state()
            
        # 군집화에 사용할 필수 특성 확인 및 생성
        required_features = ['ma_ratio_5_20', 'rsi', 'bb_width', 'macd_hist', 'volume_ratio']
        
        # 특성 추가 필요 여부 확인
        need_features = False
        for feature in required_features:
            if feature not in df.columns:
                need_features = True
                break
                
        # 특성 추가가 필요한 경우
        if need_features:
            try:
                from models.feature_engineering import FeatureEngineer
                fe = FeatureEngineer()
                
                # 기본 지표 확인
                if not all(col in df.columns for col in ['ma5', 'ma20', 'volume']):
                    logger.warning("기본 기술적 지표 누락, 추가 시도")
                    df = fe.add_indicators(df)
                
                # ML 특성 추가 시도
                df = fe.add_ml_features(df)
                
                # 필수 특성 직접 계산 (add_ml_features 실패 시)
                if 'ma_ratio_5_20' not in df.columns and 'ma5' in df.columns and 'ma20' in df.columns:
                    df['ma_ratio_5_20'] = df['ma5'] / df['ma20']
                    
                if 'volume_ratio' not in df.columns and 'volume' in df.columns:
                    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
                    
                if 'bb_width' not in df.columns and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] if 'bb_middle' in df.columns else 0
                    
            except Exception as e:
                logger.error(f"특성 생성 중 오류: {e}")
                return self._create_default_state()
        
        # 사용 가능한 특성 확인
        available_features = [f for f in required_features if f in df.columns]
        
        # 최소 2개 이상의 특성이 있어야 함
        if len(available_features) < 2:
            logger.error(f"군집화에 필요한 특성이 부족합니다. 필요: {required_features}, 가용: {available_features}")
            return self._create_default_state()
            
        try:
            # 사용 가능한 특성만 선택
            features = df[available_features].dropna()
                
            # 모델 학습 또는 로드
            if self.model is None:
                # 최근에 저장된 모델 확인
                model_path = os.path.join(self.save_dir, f"{self.ticker}_kmeans_{self.n_states}.joblib")
                if os.path.exists(model_path):
                    try:
                        # 모델 로드
                        model_data = joblib.load(model_path)
                        self.model = model_data['model']
                        self.scaler = model_data['scaler']
                        self.market_states = model_data.get('market_states', {})
                        logger.info(f"군집화 모델 로드 완료: {model_path}")
                    except Exception as e:
                        logger.error(f"모델 로드 실패: {e}")
                        self.model = None
                
                # 모델이 없으면 새로 학습
                if self.model is None:
                    # 데이터 스케일링
                    self.scaler = StandardScaler()
                    scaled_features = self.scaler.fit_transform(features)
                    
                    # KMeans 군집화
                    from sklearn.cluster import KMeans
                    self.model = KMeans(n_clusters=self.n_states, random_state=42)
                    self.model.fit(scaled_features)
                    
                    logger.info(f"새 군집화 모델 학습 완료: {self.n_states}개 군집")
            
            # 현재 상태 예측
            if len(features) > 0:
                current_features = features.iloc[-1:].values.reshape(1, -1)
                scaled_current = self.scaler.transform(current_features)
                state_id = int(self.model.predict(scaled_current)[0])
                
                # 특성 평균 계산 (해당 군집의 특성)
                if state_id in self.market_states:
                    # 기존 상태 업데이트
                    state = self.market_states[state_id]
                    features_dict = self.extract_features(df)
                    characteristics = self.characterize_market_state(features_dict)
                    state.update(features_dict, characteristics)
                else:
                    # 새 상태 생성
                    cluster_indices = (self.model.labels_ == state_id)
                    if any(cluster_indices):
                        # 해당 군집 데이터의 평균 특성 계산
                        cluster_avg = features.iloc[cluster_indices].mean().to_dict()
                        
                        # 특성 추출 및 상태 특성화
                        features_dict = self.extract_features(df)
                        characteristics = self.characterize_market_state(features_dict)
                        
                        # 새 상태 객체 생성
                        state = MarketState(state_id, features_dict, characteristics)
                        self.market_states[state_id] = state
                        
                        logger.info(f"새 시장 상태 감지: {state}")
                    else:
                        logger.warning(f"군집 {state_id}에 해당하는 데이터가 없습니다")
                        return self._create_default_state()
                
                # 상태 기록
                self.state_history.append(state_id)
                self.current_state = state
                
                # 모델 저장
                self._save_model()
                
                return state
            else:
                logger.warning("유효한 특성 데이터가 없습니다")
                return self._create_default_state()
                
        except Exception as e:
            logger.error(f"상태 감지 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_default_state()
                

    
    def _create_default_state(self):
        """기본 시장 상태 생성 (안전한 fallback)"""
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
    
    def detect_state_hmm(self, df):
        """은닉 마르코프 모델 기반 시장 상태 감지 (안전한 대안 구현)"""
        if df is None or len(df) < 20:
            logger.error("HMM을 위한 데이터 부족")
            return self._create_default_state()
            
        # hmmlearn 라이브러리 가용성 확인
        try:
            from hmmlearn import hmm
            hmmlearn_available = True
            logger.debug("hmmlearn 라이브러리 사용 가능")
        except ImportError as e:
            logger.warning(f"hmmlearn 라이브러리 없음: {e}")
            logger.info("Clustering 방법으로 대체 실행")
            hmmlearn_available = False
        except Exception as e:
            logger.error(f"hmmlearn 라이브러리 오류: {e}")
            logger.info("Clustering 방법으로 대체 실행")
            hmmlearn_available = False
        
        # hmmlearn이 없으면 clustering으로 대체
        if not hmmlearn_available:
            logger.info("HMM 대신 클러스터링 방법 사용")
            return self.detect_state_clustering(df)
        
        try:
            # HMM에 사용할 특성 선택
            required_features = ['rsi', 'bb_width', 'macd_hist', 'volume_ratio']
            available_features = [f for f in required_features if f in df.columns]
            
            if len(available_features) < 2:
                logger.warning("HMM용 특성 부족, 클러스터링으로 대체")
                return self.detect_state_clustering(df)
                
            features = df[available_features].dropna()
            if len(features) < 20:
                logger.error("유효한 특성 데이터 부족")
                return self.detect_state_clustering(df)
                
            # 스케일링
            if not hasattr(self, 'scaler') or self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                
            scaled_features = self.scaler.fit_transform(features)
            
            # HMM 모델 학습 또는 로드
            if self.model is None:
                model_path = os.path.join(self.save_dir, f"{self.ticker}_hmm_{self.n_states}.joblib")
                if os.path.exists(model_path):
                    try:
                        # 모델 로드
                        model_data = joblib.load(model_path)
                        self.model = model_data['model']
                        self.scaler = model_data['scaler']
                        self.market_states = model_data.get('market_states', {})
                        logger.info(f"HMM 모델 로드 완료: {model_path}")
                    except Exception as e:
                        logger.error(f"HMM 모델 로드 실패: {e}")
                        self.model = None
                        
                # 모델이 없으면 새로 학습
                if self.model is None:
                    try:
                        # HMM 모델 학습
                        self.model = hmm.GaussianHMM(
                            n_components=self.n_states, 
                            covariance_type="full", 
                            random_state=42,
                            n_iter=100  # 반복 횟수 제한
                        )
                        self.model.fit(scaled_features)
                        logger.info(f"새 HMM 모델 학습 완료: {self.n_states}개 상태")
                    except Exception as e:
                        logger.error(f"HMM 모델 학습 실패: {e}")
                        logger.info("클러스터링 방법으로 대체")
                        return self.detect_state_clustering(df)
            
            # 현재 상태 예측
            try:
                predicted_states = self.model.predict(scaled_features)
                current_state_id = int(predicted_states[-1])
            except Exception as e:
                logger.error(f"HMM 상태 예측 실패: {e}")
                logger.info("클러스터링 방법으로 대체")
                return self.detect_state_clustering(df)
            
            # 특성 추출 및 상태 특성화
            features_dict = self.extract_features(df)
            characteristics = self.characterize_market_state(features_dict)
            
            # 상태 객체 업데이트 또는 생성
            if current_state_id in self.market_states:
                state = self.market_states[current_state_id]
                state.update(features_dict, characteristics)
            else:
                state = MarketState(current_state_id, features_dict, characteristics)
                self.market_states[current_state_id] = state
                logger.info(f"새 시장 상태 감지: {state}")
            
            # 상태 기록
            self.state_history.append(current_state_id)
            self.current_state = state
            
            # 모델 저장
            self._save_model()
            
            return state
            
        except Exception as e:
            logger.error(f"HMM 상태 감지 중 예외 발생: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            logger.info("클러스터링 방법으로 대체 실행")
            return self.detect_state_clustering(df)
    
    def detect_state_cusum(self, df):
        """CUSUM 알고리즘 기반 시장 체제 변화 감지"""
        if df is None or len(df) < 10:
            logger.error("CUSUM을 위한 데이터 부족")
            return None
            
        # 감시할 시계열 (종가 또는 다른 지표)
        target_series = df['close']
        current_value = target_series.iloc[-1]
        
        # 기준값 초기화 (첫 실행 시)
        if self.cusum_baseline is None:
            self.cusum_baseline = target_series.iloc[-5:].mean()  # 최근 5개 평균
            self.cusum_pos = 0
            self.cusum_neg = 0
            logger.info(f"CUSUM 기준값 초기화: {self.cusum_baseline:.2f}")
        
        # 표준편차 계산
        std_dev = target_series.pct_change().std()
        
        # 현재값과 기준값의 차이 계산
        diff = current_value / self.cusum_baseline - 1  # 수익률로 계산
        
        # CUSUM 업데이트
        self.cusum_pos = max(0, self.cusum_pos + diff)
        self.cusum_neg = max(0, self.cusum_neg - diff)
        
        # 변화 감지 여부 확인
        state_changed = False
        threshold = self.cusum_threshold * std_dev
        
        if self.cusum_pos > threshold:
            logger.info(f"CUSUM: 상승 추세 변화 감지 (누적값: {self.cusum_pos:.4f})")
            self.cusum_baseline = current_value  # 기준값 업데이트
            self.cusum_pos = 0
            self.cusum_neg = 0
            state_changed = True
            
        elif self.cusum_neg > threshold:
            logger.info(f"CUSUM: 하락 추세 변화 감지 (누적값: {self.cusum_neg:.4f})")
            self.cusum_baseline = current_value  # 기준값 업데이트
            self.cusum_pos = 0
            self.cusum_neg = 0
            state_changed = True
        
        # 상태 변화가 감지되면 새 상태 생성
        if state_changed:
            # 상태 ID 생성 (현재 시간 기반)
            state_id = datetime.now().strftime("%Y%m%d%H%M")
            
            # 특성 추출 및 상태 특성화
            features = self.extract_features(df)
            characteristics = self.characterize_market_state(features)
            
            # 새 상태 객체 생성
            state = MarketState(state_id, features, characteristics)
            self.market_states[state_id] = state
            
            # 상태 기록
            self.state_history.append(state_id)
            self.current_state = state
            
            logger.info(f"새 시장 상태 감지: {state}")
            return state
        
        # 변화가 없으면 현재 상태 유지
        if not self.current_state:
            # 초기 상태 생성
            state_id = datetime.now().strftime("%Y%m%d%H%M")
            features = self.extract_features(df)
            characteristics = self.characterize_market_state(features)
            self.current_state = MarketState(state_id, features, characteristics)
            self.market_states[state_id] = self.current_state
            self.state_history.append(state_id)
            
        return self.current_state
    
    def detect_current_state(self):
        """현재 시장 상태 감지 (안전한 fallback 포함)"""
        # 이전 상태 저장
        if self.current_state:
            self.previous_state_id = self.current_state.state_id
        
        # 데이터 수집
        df = self.collect_data()
        
        if df is None or len(df) < 10:
            logger.warning("데이터 부족으로 기본 상태 사용")
            return self._create_default_state()
        
        # 감지 방법에 따라 다른 알고리즘 사용 (안전한 순서)
        new_state = None
        
        try:
            if self.detection_method == 'clustering':
                new_state = self.detect_state_clustering(df)
            elif self.detection_method == 'hmm':
                new_state = self.detect_state_hmm(df)
            elif self.detection_method == 'cusum':
                new_state = self.detect_state_cusum(df)
            else:
                logger.error(f"지원하지 않는 감지 방법: {self.detection_method}")
                new_state = self.detect_state_clustering(df)  # 기본값으로 clustering 사용
                
        except Exception as e:
            logger.error(f"상태 감지 중 예외 발생: {e}")
            logger.info("기본 클러스터링 방법으로 대체")
            try:
                new_state = self.detect_state_clustering(df)
            except Exception as e2:
                logger.error(f"클러스터링도 실패: {e2}")
                new_state = self._create_default_state()
        
        # 최종 안전장치
        if new_state is None:
            logger.warning("모든 상태 감지 방법 실패, 기본 상태 사용")
            new_state = self._create_default_state()
        
        # 상태가 변경된 경우에만 previous_state_id 업데이트 (수정)
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
        
        # 변동성 기반 전략 조정
        if 'volatility' in characteristics:
            if characteristics['volatility'] in ["매우 높음", "높음"]:
                if 'params' not in strategy_info:
                    strategy_info['params'] = {}
                strategy_info['params']['stop_loss'] = 0.04  # 변동성 높을 때 손절 타이트하게
                strategy_info['params']['take_profit'] = 0.08  # 변동성 높을 때 익절 적극적으로
            elif characteristics['volatility'] == "낮음":
                if 'params' not in strategy_info:
                    strategy_info['params'] = {}
                strategy_info['params']['stop_loss'] = 0.07  # 변동성 낮을 때 손절 여유있게
                strategy_info['params']['take_profit'] = 0.12  # 변동성 낮을 때 익절 여유있게
        
        # 상태 업데이트
        state.optimal_strategy = strategy_info
        
        return strategy_info
    
    def update_state_performance(self, state_id, strategy_name, performance):
        """상태별 전략 성능 업데이트"""
        if state_id not in self.market_states:
            logger.warning(f"알 수 없는 상태 ID: {state_id}")
            return False
            
        # 상태-전략 성능 기록
        if state_id not in self.state_metrics:
            self.state_metrics[state_id] = {}
            
        if strategy_name not in self.state_metrics[state_id]:
            self.state_metrics[state_id][strategy_name] = []
            
        # 성능 추가
        self.state_metrics[state_id][strategy_name].append(performance)
        
        # 성능에 따라 최적 전략 업데이트
        self._update_optimal_strategy(state_id)
        
        return True
    
    def _update_optimal_strategy(self, state_id):
        """성능 기반으로 최적 전략 업데이트"""
        if state_id not in self.state_metrics:
            return
        
        best_strategy = None
        best_performance = -float('inf')
        
        # 각 전략별 평균 성능 계산
        for strategy_name, performances in self.state_metrics[state_id].items():
            if not performances:
                continue
                
            avg_performance = sum(performances) / len(performances)
            
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_strategy = strategy_name
        
        # 최적 전략 업데이트
        if best_strategy and state_id in self.market_states:
            self.market_states[state_id].optimal_strategy = {
                'type': best_strategy,
                'avg_performance': best_performance
            }
            logger.info(f"상태 {state_id}의 최적 전략 업데이트: {best_strategy} (성능: {best_performance:.2f})")
    
    def _save_model(self):
        """현재 모델 및 상태 정보 저장"""
        try:
            # 저장 파일명 생성
            filename = f"{self.ticker}_{self.detection_method}_{self.n_states}.joblib"
            filepath = os.path.join(self.save_dir, filename)
            
            # 저장 데이터 준비
            save_data = {
                'model': self.model,
                'scaler': self.scaler,
                'detection_method': self.detection_method,
                'n_states': self.n_states,
                'market_states': self.market_states,
                'state_metrics': self.state_metrics,
                'saved_at': datetime.now()
            }
            
            # 저장
            joblib.dump(save_data, filepath)
            logger.debug(f"모델 저장 완료: {filepath}")
            
            # 상태 정보 JSON으로도 저장 (읽기 쉽게)
            json_path = os.path.join(self.save_dir, f"{self.ticker}_states.json")
            
            states_json = {}
            for state_id, state in self.market_states.items():
                states_json[state_id] = {
                    'characteristics': state.characteristics,
                    'occurrences': state.occurrences,
                    'last_seen': state.last_seen.strftime("%Y-%m-%d %H:%M:%S"),
                    'optimal_strategy': state.optimal_strategy
                }
                
            with open(json_path, 'w') as f:
                json.dump(states_json, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return False
    
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
                    logger.info(f"현재 시장 상태: {current_state}")
                
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

# 사용 예시
if __name__ == "__main__":
    detector = MarketStateDetector(ticker="KRW-BTC", detection_method='clustering', n_states=5)
    
    # 자동 감지 시작
    detector.start_auto_detection()
    
    try:
        # 메인 스레드는 다른 작업 수행 가능
        while True:
            time.sleep(10)
            if detector.current_state:
                print(f"현재 상태: {detector.current_state}")
                print(f"최적 전략: {detector.get_optimal_strategy()}")
                print("------------------------")
    
    except KeyboardInterrupt:
        detector.stop_auto_detection()
        print("프로그램이 종료되었습니다.")
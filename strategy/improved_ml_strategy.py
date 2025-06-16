
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import logging
from utils.logger import setup_logger

logger = setup_logger("improved_ml_strategy")

class ImprovedMLStrategy:
    """개선된 ML 전략 - 과적합 방지 및 앙상블"""
    
    def __init__(self, ticker, min_data_points=200):
        self.ticker = ticker
        self.min_data_points = min_data_points
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.last_performance = {}
        
    def prepare_features(self, df):
        """특성 준비 - 더 안정적인 특성 선택"""
        if len(df) < self.min_data_points:
            logger.warning(f"{self.ticker}: 데이터 부족 ({len(df)}/{self.min_data_points})")
            return None, None
            
        try:
            features = pd.DataFrame(index=df.index)
            
            # 가격 기반 특성
            features['price_ma_5'] = df['close'].rolling(5).mean() / df['close']
            features['price_ma_20'] = df['close'].rolling(20).mean() / df['close']
            features['price_ma_60'] = df['close'].rolling(60).mean() / df['close']
            
            # 변동성 특성
            features['volatility_5'] = df['close'].pct_change().rolling(5).std()
            features['volatility_20'] = df['close'].pct_change().rolling(20).std()
            
            # 거래량 특성
            volume_col = 'volume' if 'volume' in df.columns else 'candle_acc_trade_volume'
            features['volume_ma_5'] = df[volume_col].rolling(5).mean()
            features['volume_ma_20'] = df[volume_col].rolling(20).mean()
            features['volume_ratio'] = df[volume_col] / features['volume_ma_20']
            
            # RSI (14일)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # 타겟 변수 (다음날 수익률 기준)
            future_returns = df['close'].shift(-1) / df['close'] - 1
            target = np.where(future_returns > 0.02, 2,  # 강한 상승
                            np.where(future_returns > 0.005, 1,  # 약한 상승
                                   np.where(future_returns < -0.02, -2,  # 강한 하락
                                          np.where(future_returns < -0.005, -1, 0))))  # 약한 하락, 중립
            
            # NaN 제거
            valid_mask = features.notna().all(axis=1) & ~np.isnan(target)
            features_clean = features[valid_mask].fillna(0)
            target_clean = target[valid_mask]
            
            if len(features_clean) < self.min_data_points // 2:
                logger.warning(f"{self.ticker}: 유효 데이터 부족")
                return None, None
                
            return features_clean, target_clean
            
        except Exception as e:
            logger.error(f"{self.ticker} 특성 준비 오류: {e}")
            return None, None
    
    def train_ensemble_model(self, features, target):
        """앙상블 모델 훈련"""
        try:
            # 시계열 분할
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 여러 모델 정의
            models = {
                'rf': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=20,
                    random_state=42
                )
            }
            
            # 각 모델 훈련 및 검증
            trained_models = {}
            for name, model in models.items():
                # 교차 검증
                cv_scores = cross_val_score(model, features, target, cv=tscv, scoring='accuracy')
                logger.info(f"{self.ticker} {name} CV 점수: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                
                # 전체 데이터로 훈련
                model.fit(features, target)
                trained_models[name] = model
                
                # 특성 중요도 저장
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(features.columns, model.feature_importances_))
            
            self.models = trained_models
            
            # 성능 기록
            self.last_performance = {
                'train_samples': len(features),
                'feature_count': len(features.columns),
                'cv_scores': {name: cross_val_score(model, features, target, cv=tscv, scoring='accuracy').mean() 
                            for name, model in trained_models.items()}
            }
            
            logger.info(f"{self.ticker} 앙상블 모델 훈련 완료")
            return True
            
        except Exception as e:
            logger.error(f"{self.ticker} 모델 훈련 오류: {e}")
            return False
    
    def predict_ensemble(self, features):
        """앙상블 예측"""
        if not self.models:
            return 0  # 중립
            
        try:
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                pred = model.predict(features)[0]
                # CV 점수를 가중치로 사용
                weight = self.last_performance.get('cv_scores', {}).get(name, 0.5)
                
                predictions.append(pred)
                weights.append(weight)
            
            # 가중 평균 (소수점은 반올림)
            weighted_pred = np.average(predictions, weights=weights)
            final_pred = int(round(weighted_pred))
            
            # 신뢰도 계산 (예측 일치도)
            confidence = 1.0 - (np.std(predictions) / 2.0)  # 표준편차 기반
            
            # 낮은 신뢰도에서는 중립으로
            if confidence < 0.6:
                final_pred = 0
                
            return final_pred
            
        except Exception as e:
            logger.error(f"{self.ticker} 예측 오류: {e}")
            return 0
    
    def get_signal(self, df):
        """거래 신호 생성"""
        try:
            features, _ = self.prepare_features(df)
            if features is None:
                return 'hold'
                
            # 최신 데이터로 예측
            last_features = features.iloc[-1:].fillna(0)
            prediction = self.predict_ensemble(last_features)
            
            # 더 보수적인 신호 생성
            if prediction >= 2:
                return 'strong_buy'
            elif prediction >= 1:
                return 'buy'
            elif prediction <= -2:
                return 'strong_sell'
            elif prediction <= -1:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            logger.error(f"{self.ticker} 신호 생성 오류: {e}")
            return 'hold'

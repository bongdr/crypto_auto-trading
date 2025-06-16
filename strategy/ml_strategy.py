import pandas as pd
import numpy as np
import os
import joblib
from strategy.base import BaseStrategy
from utils.logger import setup_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

logger = setup_logger("ml_strategy")

class MLStrategy(BaseStrategy):
    """머신러닝 기반 매매 전략 - 개선된 버전"""
    
    def __init__(self, name="머신러닝 전략", model_path=None, model_type='random_forest'):
        super().__init__(name)
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.model_type = model_type
        self.feature_cols = []
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def prepare_features(self, df):
        """모델 학습/예측을 위한 특성 준비"""
        required_indicators = ['ma5', 'ma20', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'volume_ratio']
        if not all(ind in df.columns for ind in required_indicators):
            logger.warning("필요한 지표가 데이터에 없습니다. 기술적 지표를 먼저 추가해주세요.")
            return None
            
        features = pd.DataFrame(index=df.index)
        features['ma_ratio_5_20'] = df['ma5'] / df['ma20']
        features['price_to_ma20'] = df['close'] / df['ma20']
        features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        features['rsi'] = df['rsi']
        features['macd_hist'] = df['macd'] - df['macd_signal'] if 'macd_signal' in df.columns else df['macd']
        features['volume_ratio'] = df['volume_ratio']
        
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        self.feature_cols = features.columns.tolist()
        return features
    
    def prepare_target(self, df, horizon=3, threshold=0.01):
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df['target'] = (df['future_return'] > threshold).astype(int)
        return df['target']
    
    def train_model(self, df, horizon=3, threshold=0.01, test_size=0.2, save_path=None):
        """모델 학습 - 과적합 완화"""
        from sklearn.model_selection import train_test_split
        
        features = self.prepare_features(df)
        if features is None:
            return False
            
        target = self.prepare_target(df, horizon, threshold)
        common_index = features.index.intersection(target.dropna().index)
        X = features.loc[common_index]
        y = target.loc[common_index]
        
        min_data_count = max(100, len(X) * 0.8)
        if len(X) < min_data_count:
            logger.warning(f"학습 데이터가 부족합니다: {len(X)} 행 (최소 {min_data_count} 필요)")
            return False
            
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        param_grid = {
            'n_estimators': [30, 50],
            'max_depth': [3, 5],
            'min_samples_split': [10],
            'min_samples_leaf': [5]
        }
        base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.model = GridSearchCV(base_model, param_grid, cv=3, scoring='f1')
        
        try:
            self.model.fit(X_train_scaled, y_train)
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test) if len(X_test) > 0 else train_accuracy
            self.test_accuracy = test_accuracy
            
            logger.info(f"모델 학습 완료: 훈련 정확도 {train_accuracy:.4f}, 테스트 정확도 {test_accuracy:.4f}")
            
            if test_accuracy < 0.75:
                logger.warning(f"테스트 정확도 {test_accuracy:.4f}가 낮습니다. 기본 전략으로 전환")
                return False
                
            if hasattr(self.model.best_estimator_, 'feature_importances_'):
                importances = self.model.best_estimator_.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': self.feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                self.feature_cols = feature_importance[feature_importance['Importance'] > 0.1]['Feature'].tolist()
                logger.debug("특성 중요도 상위 3개:")
                for i, row in feature_importance.head(3).iterrows():
                    logger.debug(f"  {row['Feature']}: {row['Importance']:.4f}")
            
            if save_path:
                self.save_model(save_path)
            return True
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류: {e}")
            return False
    
    def predict_signals(self, features):
        """특성 데이터로 신호 예측"""
        if self.model is None:
            logger.error("모델이 학습되지 않았습니다")
            return np.zeros(len(features))
            
        required_features = set(self.feature_cols)
        available_features = set(features.columns)
        missing_features = required_features - available_features
        
        if missing_features:
            logger.warning(f"누락된 특성 감지: {missing_features}")
            features_copy = features.copy()
            
            for feature in missing_features:
                if feature == 'ma_ratio_5_20' and 'ma5' in features.columns and 'ma20' in features.columns:
                    features_copy[feature] = features['ma5'] / features['ma20']
                elif feature == 'price_to_ma20' and 'close' in features.columns and 'ma20' in features.columns:
                    features_copy[feature] = features['close'] / features['ma20']
                elif feature == 'bb_position' and 'close' in features.columns and 'bb_upper' in features.columns and 'bb_lower' in features.columns:
                    features_copy[feature] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
                elif feature == 'macd_hist' and 'macd' in features.columns and 'macd_signal' in features.columns:
                    features_copy[feature] = features['macd'] - features['macd_signal']
                elif feature == 'volume_ratio' and 'volume' in features.columns:
                    features_copy[feature] = features['volume'] / features['volume'].rolling(window=20).mean()
                else:
                    logger.warning(f"특성 '{feature}' 자동 계산 불가능, 중간값으로 대체")
                    features_copy[feature] = features[feature].median() if feature in features else 0.5
            
            prediction_features = features_copy[list(required_features)]
        else:
            prediction_features = features[self.feature_cols]
        
        try:
            prediction_features = prediction_features.fillna(method='ffill').fillna(method='bfill')
            
            for col in prediction_features.columns:
                z_scores = (prediction_features[col] - prediction_features[col].mean()) / prediction_features[col].std()
                outliers = abs(z_scores) > 3
                if outliers.any():
                    median_val = prediction_features[col].median()
                    prediction_features.loc[outliers, col] = median_val
            
            scaled_features = self.scaler.transform(prediction_features)
            probabilities = self.model.predict_proba(scaled_features)
            buy_probabilities = probabilities[:, 1]
            return buy_probabilities
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros(len(features))
    
    def generate_signal(self, df, probability_threshold=None, market_state=None):
        """매매 신호 생성 - 동적 임계값"""
        features = self.prepare_features(df)
        if features is None or self.model is None:
            return pd.Series(0, index=df.index)

        if probability_threshold is None:
            atr = (df['high'] - df['low']).rolling(window=20).mean() / df['close']
            volatility = atr.iloc[-1] if not atr.empty else 0.01
            probability_threshold = max(0.45, min(0.55, 0.45 + volatility * 2))
            
            if market_state and 'trend' in market_state.characteristics:
                if market_state.characteristics['trend'].startswith('강한'):
                    probability_threshold = min(probability_threshold + 0.05, 0.6)
                else:
                    probability_threshold = max(probability_threshold - 0.05, 0.4)

        common_index = features.index.intersection(df.index)
        signals = pd.Series(0, index=df.index)

        if len(common_index) > 0:
            buy_probabilities = self.predict_signals(features.loc[common_index])
            signals_subset = pd.Series(0, index=common_index)
            signals_subset[buy_probabilities > probability_threshold] = 1
            signals_subset[buy_probabilities < (1 - probability_threshold - 0.05)] = -1
            signals.loc[common_index] = signals_subset

        return signals
    
    def save_model(self, path):
        if self.model is None:
            logger.error("저장할 모델이 없습니다")
            return False
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model.best_estimator_ if hasattr(self.model, 'best_estimator_') else self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'test_accuracy': getattr(self, 'test_accuracy', 0.5)
        }
        joblib.dump(model_data, path)
        logger.info(f"모델 저장 완료: {path}")
        return True
    
    def load_model(self, path):
        if not os.path.exists(path):
            logger.error(f"모델 파일이 존재하지 않습니다: {path}")
            return False
            
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_cols = model_data['feature_cols']
            self.model_type = model_data.get('model_type', 'random_forest')
            self.test_accuracy = model_data.get('test_accuracy', 0.5)
            logger.info(f"모델 로드 완료: {path}")
            return True
        except Exception as e:
            logger.error(f"모델 로드 오류: {e}")
            return False

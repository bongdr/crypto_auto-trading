
#!/usr/bin/env python3
"""가상화폐 자동거래 시스템 수정 스크립트"""
import os
import shutil
from pathlib import Path

def create_backup(file_path):
    """파일 백업 생성"""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(file_path, backup_path)
        print(f"백업 생성: {backup_path}")

def update_ml_strategy():
    """ml_strategy.py 수정"""
    file_path = "strategy/ml_strategy.py"
    create_backup(file_path)
    
    content = '''import pandas as pd
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
'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ {file_path} 수정 완료")

def update_trading_executor():
    """trading_executor.py 수정"""
    file_path = "trading/execution.py"
    create_backup(file_path)
    
    content = '''from config.settings import TRADING_MODE
from trading.paper_account import PaperAccount
from trading.order_manager import OrderManager
from trading.risk_management import RiskManager
from data.collector import UpbitDataCollector
from utils.logger import setup_logger
from models.feature_engineering import FeatureEngineer
import pandas as pd
import numpy as np
import pyupbit
import time
from datetime import datetime

logger = setup_logger("trading_executor")

class TradingExecutor:
    """거래 실행 관리"""
    
    def __init__(self, strategy, initial_balance=1000000):
        self.strategy = strategy
        self.collector = UpbitDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager()
        
        if TRADING_MODE == "paper":
            self.paper_account = PaperAccount(initial_balance)
            self.order_manager = OrderManager(self.paper_account)
            logger.info(f"페이퍼 트레이딩 모드로 초기화 (초기 잔고: {initial_balance})")
        else:
            self.order_manager = OrderManager()
            logger.info("실제 거래 모드로 초기화")
            
        self.running = False
        self.current_data = {}
        self.positions = {}
        self.position_info = {}
    
    def start_trading(self, tickers):
        if isinstance(tickers, str):
            tickers = [tickers]
            
        self.tickers = tickers
        self.running = True
        
        for ticker in tickers:
            amount = self.order_manager.get_balance(ticker)
            if amount > 0:
                self.positions[ticker] = 1
                avg_price = self.order_manager.get_avg_buy_price(ticker)
                self.position_info[ticker] = {
                    'entry_price': avg_price,
                    'entry_time': datetime.now(),
                    'amount': amount
                }
            else:
                self.positions[ticker] = 0
                
        logger.info(f"거래 시작: {', '.join(tickers)}")
        
        for ticker in tickers:
            self.update_data(ticker)
            
        return True
        
    def stop_trading(self):
        self.running = False
        logger.info("거래 중지")
        
        if TRADING_MODE == "paper":
            current_prices = {}
            for ticker in self.tickers:
                current_price = self.get_current_price(ticker)
                if current_price:
                    current_prices[ticker] = current_price
            
            portfolio_value = self.paper_account.get_portfolio_value(current_prices)
            initial_balance = self.paper_account.initial_balance
            
            logger.info(f"===== 거래 결과 요약 =====")
            logger.info(f"초기 자본: {initial_balance:,.0f}원")
            logger.info(f"최종 포트폴리오: {portfolio_value:,.0f}원")
            logger.info(f"수익률: {((portfolio_value / initial_balance) - 1) * 100:.2f}%")
            logger.info(f"현금 잔고: {self.paper_account.balance:,.0f}원")
            
            for ticker, amount in self.paper_account.holdings.items():
                if amount > 0:
                    value = amount * current_prices.get(ticker, 0)
                    avg_price = self.paper_account.buy_prices.get(ticker, 0)
                    profit_percent = ((current_prices.get(ticker, 0) / avg_price) - 1) * 100 if avg_price > 0 else 0
                    logger.info(f"{ticker}: {amount:.8f}개, 평균매수가 {avg_price:,.0f}원, 현재가 {current_prices.get(ticker, 0):,.0f}원, 평가금액 {value:,.0f}원, 수익률 {profit_percent:.2f}%")
                    
            logger.info("==========================")
            
        return True
        
    def update_data(self, ticker):
        max_retries = 5
        retry_count = 0
        df = None
        
        while retry_count < max_retries and df is None:
            try:
                df = self.collector.get_ohlcv(ticker, interval="minute10", count=100)
                retry_count += 1
                if df is None and retry_count < max_retries:
                    time.sleep(2)
            except Exception as e:
                logger.error(f"데이터 수집 오류 (재시도 {retry_count}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)
        
        if df is not None:
            try:
                df = self.feature_engineer.add_indicators(df)
                current_price = self.get_current_price(ticker)
                
                if current_price:
                    self.current_data[ticker] = {
                        'df': df,
                        'current_price': current_price,
                        'updated_at': pd.Timestamp.now()
                    }
                    logger.debug(f"데이터 업데이트: {ticker}, 현재가: {current_price}")
                    return True
                else:
                    logger.error(f"현재가 조회 실패: {ticker}")
                    return False
            except Exception as e:
                logger.error(f"지표 생성 오류: {ticker}, {e}")
                return False
        else:
            logger.error(f"데이터 업데이트 실패: {ticker}, 최대 재시도 횟수 초과")
            return False
    
    def check_signals(self, ticker):
        if ticker not in self.current_data:
            logger.warning(f"데이터 없음: {ticker}")
            return None
            
        data = self.current_data[ticker]
        df = data['df']
        
        signal = self.strategy.generate_signal(df)
        
        if len(signal) > 0:
            last_signal = signal.iloc[-1]
            current_position = self.positions.get(ticker, 0)
            current_price = data.get('current_price')
            
            if current_price is None:
                logger.error(f"{ticker} 현재가 데이터 없음")
                return None
                
            return {
                'ticker': ticker,
                'signal': last_signal,
                'current_position': current_position,
                'current_price': current_price
            }
        
        return None
        
    def execute_signal(self, signal_info):
        if signal_info is None:
            logger.warning("신호 정보가 없습니다")
            return {'action': 'none', 'reason': 'no_signal_info'}
            
        ticker = signal_info.get('ticker')
        if ticker is None:
            logger.warning("신호 정보에 티커가 없습니다")
            return {'action': 'none', 'reason': 'no_ticker'}
            
        signal = signal_info.get('signal', 0)
        current_position = signal_info.get('current_position', 0)
        current_price = float(signal_info.get('current_price', 0))
        
        if current_price <= 0:
            logger.warning(f"{ticker} 현재가 없음, 거래 무시")
            return {'action': 'none', 'ticker': ticker, 'reason': 'no_price'}
        
        risk_check = self.risk_manager.check_trade(
            ticker=ticker,
            price=current_price,
            position=current_position,
            position_info=self.position_info.get(ticker),
            signal=signal,
            account_info=self.paper_account if TRADING_MODE == "paper" else None
        )
        
        if not risk_check['allow_trade']:
            logger.info(f"리스크 관리에 의해 거래 제한: {ticker}, 이유: {risk_check['reason']}")
            return {'action': 'none', 'ticker': ticker, 'reason': risk_check['reason']}
        
        if signal == 1 and current_position == 0:
            balance = self.order_manager.get_balance("KRW")
            confirmed_price = self.get_current_price(ticker)
            
            if confirmed_price is None:
                logger.warning(f"{ticker} 매수 직전 현재가 재확인 실패, 거래 취소")
                return {'action': 'none', 'ticker': ticker, 'reason': 'price_recheck_failed'}
            
            position_size = risk_check.get('position_size', 0.3)
            buy_amount = balance * position_size
            
            if buy_amount >= 5000:
                order = self.order_manager.buy_market_order(ticker, buy_amount)
                
                if order:
                    self.positions[ticker] = 1
                    entry_amount = buy_amount / current_price
                    self.position_info[ticker] = {
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'amount': entry_amount
                    }
                    
                    logger.info(f"매수 실행: {ticker}, 금액: {buy_amount:,.0f}원, 가격: {current_price:,.0f}원")
                    
                    if hasattr(self, 'telegram_notifier') and self.telegram_notifier:
                        trade_info = {
                            'action': 'buy',
                            'ticker': ticker,
                            'price': current_price,
                            'amount': entry_amount,
                            'total': buy_amount
                        }
                        self.telegram_notifier.notify_trade(trade_info)
                    
                    return {'action': 'buy', 'ticker': ticker, 'amount': buy_amount, 'price': current_price}
                else:
                    logger.error(f"매수 주문 실패: {ticker}")
            else:
                logger.warning(f"매수 금액 부족: {buy_amount} < 5000원")
        
        elif signal == -1 and current_position == 1:
            coin_amount = self.order_manager.get_balance(ticker)
            
            if coin_amount > 0:
                order = self.order_manager.sell_market_order(ticker, coin_amount)
                
                if order:
                    self.positions[ticker] = 0
                    entry_price = self.position_info.get(ticker, {}).get('entry_price', 0)
                    profit_percent = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                    
                    logger.info(f"매도 실행: {ticker}, 수량: {coin_amount}, 가격: {current_price:,.0f}원, 수익률: {profit_percent:.2f}%")
                    
                    if hasattr(self, 'telegram_notifier') and self.telegram_notifier:
                        trade_info = {
                            'action': 'sell',
                            'ticker': ticker,
                            'price': current_price,
                            'amount': coin_amount,
                            'profit_percent': profit_percent
                        }
                        self.telegram_notifier.notify_trade(trade_info)
                    
                    self.position_info.pop(ticker, None)
                    return {'action': 'sell', 'ticker': ticker, 'amount': coin_amount, 'price': current_price, 'profit_percent': profit_percent}
                else:
                    logger.error(f"매도 주문 실패: {ticker}")
            else:
                logger.warning(f"매도할 코인 없음: {ticker}")
                
        elif current_position == 1:
            position_info = self.position_info.get(ticker)
            if position_info:
                entry_price = position_info.get('entry_price', 0)
                
                stop_loss = risk_check.get('stop_loss')
                if stop_loss and current_price <= stop_loss:
                    coin_amount = self.order_manager.get_balance(ticker)
                    order = self.order_manager.sell_market_order(ticker, coin_amount)
                    
                    if order:
                        self.positions[ticker] = 0
                        loss_percent = ((current_price / entry_price) - 1) * 100
                        logger.info(f"손절 매도: {ticker}, 가격: {current_price:,.0f}원, 손실률: {loss_percent:.2f}%")
                        self.position_info.pop(ticker, None)
                        return {'action': 'stop_loss', 'ticker': ticker, 'amount': coin_amount, 'price': current_price}
                
                take_profit = risk_check.get('take_profit')
                if take_profit and current_price >= take_profit:
                    coin_amount = self.order_manager.get_balance(ticker)
                    order = self.order_manager.sell_market_order(ticker, coin_amount)
                    
                    if order:
                        self.positions[ticker] = 0
                        profit_percent = ((current_price / entry_price) - 1) * 100
                        logger.info(f"익절 매도: {ticker}, 가격: {current_price:,.0f}원, 수익률: {profit_percent:.2f}%")
                        self.position_info.pop(ticker, None)
                        return {'action': 'take_profit', 'ticker': ticker, 'amount': coin_amount, 'price': current_price}
        
        return {'action': 'none', 'ticker': ticker}
    
    def update_and_trade(self, ticker):
        if not self.running:
            return False
            
        updated = self.update_data(ticker)
        
        if not updated:
            return False
            
        signal_info = self.check_signals(ticker)
        
        if signal_info:
            result = self.execute_signal(signal_info)
            return result
            
        return {'action': 'none', 'ticker': ticker}
    
    def run_trading_loop(self, interval=60):
        logger.info(f"거래 루프 시작: {interval}초 간격")
        
        try:
            while self.running:
                for ticker in self.tickers:
                    self.update_and_trade(ticker)
                
                if TRADING_MODE == "paper":
                    current_prices = {}
                    for ticker in self.tickers:
                        current_price = self.get_current_price(ticker)
                        if current_price:
                            current_prices[ticker] = current_price
                    
                    portfolio_value = self.paper_account.get_portfolio_value(current_prices)
                    initial_balance = self.paper_account.initial_balance
                    profit_percent = ((portfolio_value / initial_balance) - 1) * 100
                    
                    logger.info(f"현재 포트폴리오 가치: {portfolio_value:,.0f}원 (수익률: {profit_percent:.2f}%)")
                    
                    self.order_manager.update_limit_orders()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("사용자에 의한 중지")
            self.stop_trading()
            
        except Exception as e:
            logger.error(f"거래 루프 오류: {e}")
            self.stop_trading()
    
    def get_multi_timeframe_data(self, ticker, timeframes):
        result = {}
        
        for tf in timeframes:
            if tf == 'day':
                interval = 'day'
                count = 100
            elif tf == 'hour4':
                interval = 'minute240'
                count = 100
            elif tf == 'hour':
                interval = 'minute60'
                count = 100
            elif tf == 'minute30':
                interval = 'minute30'
                count = 100
            else:
                logger.warning(f"지원하지 않는 시간프레임: {tf}")
                continue
                
            df = self.collector.get_ohlcv(ticker, interval=interval, count=count)
            
            if df is not None and hasattr(self.feature_engineer, 'add_indicators'):
                df = self.feature_engineer.add_indicators(df)
                
            result[tf] = df
                
        return result
        
    def get_current_price(self, ticker):
        for attempt in range(5):
            price = pyupbit.get_current_price(ticker)
            if price:
                return price
            time.sleep(2)
        logger.error(f"{ticker} 현재가 조회 실패")
        return self.current_data.get(ticker, {}).get('current_price')
'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ {file_path} 수정 완료")

def update_order_manager():
    """order_manager.py 수정"""
    file_path = "trading/order_manager.py"
    create_backup(file_path)
    
    content = '''from config.settings import TRADING_MODE, UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
from utils.logger import setup_logger
import pyupbit
import time

logger = setup_logger("order_manager")

class OrderManager:
    """주문 관리"""
    
    def __init__(self, paper_account=None):
        self.mode = TRADING_MODE
        
        if self.mode == "live":
            try:
                self.upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
                logger.info("실제 거래 모드로 초기화")
            except Exception as e:
                logger.error(f"업비트 연결 오류: {e}")
                self.upbit = None
        else:
            self.paper_account = paper_account
            logger.info("페이퍼 트레이딩 모드로 초기화")
            
    def buy_market_order(self, ticker, amount):
        for attempt in range(3):
            try:
                if self.mode == "live":
                    if self.upbit is None:
                        logger.error("업비트 연결이 없습니다")
                        return None
                        
                    order = self.upbit.buy_market_order(ticker, amount)
                    logger.info(f"실제 매수 주문: {ticker}, 금액: {amount}")
                    return order
                else:
                    current_price = pyupbit.get_current_price(ticker)
                    
                    if current_price is None:
                        logger.error(f"현재가 조회 실패: {ticker}")
                        return None
                        
                    quantity = amount / current_price
                    success = self.paper_account.buy(ticker, current_price, quantity)
                    
                    if success:
                        logger.info(f"가상 매수 주문: {ticker}, 가격: {current_price}, 수량: {quantity}")
                        return {'success': True, 'ticker': ticker, 'price': current_price, 'amount': quantity}
                    else:
                        logger.warning(f"가상 매수 실패: {ticker}")
                        return None
            except Exception as e:
                logger.error(f"매수 시도 {attempt+1} 실패: {e}")
                if attempt < 2:
                    time.sleep(2)
        return None
        
    def sell_market_order(self, ticker, quantity):
        for attempt in range(3):
            try:
                if self.mode == "live":
                    if self.upbit is None:
                        logger.error("업비트 연결이 없습니다")
                        return None
                        
                    order = self.upbit.sell_market_order(ticker, quantity)
                    logger.info(f"실제 매도 주문: {ticker}, 수량: {quantity}")
                    return order
                else:
                    current_price = pyupbit.get_current_price(ticker)
                    
                    if current_price is None:
                        logger.error(f"현재가 조회 실패: {ticker}")
                        return None
                        
                    success = self.paper_account.sell(ticker, current_price, quantity)
                    
                    if success:
                        logger.info(f"가상 매도 주문: {ticker}, 가격: {current_price}, 수량: {quantity}")
                        return {'success': True, 'ticker': ticker, 'price': current_price, 'amount': quantity}
                    else:
                        logger.warning(f"가상 매도 실패: {ticker}")
                        return None
            except Exception as e:
                logger.error(f"매도 시도 {attempt+1} 실패: {e}")
                if attempt < 2:
                    time.sleep(2)
        return None
    
    def get_balance(self, ticker="KRW"):
        try:
            if self.mode == "live":
                if self.upbit is None:
                    logger.error("업비트 연결이 없습니다")
                    return 0
                    
                return self.upbit.get_balance(ticker)
            else:
                if ticker == "KRW":
                    return self.paper_account.get_balance()
                else:
                    holdings = self.paper_account.get_holdings()
                    return holdings.get(ticker, 0)
        except Exception as e:
            logger.error(f"잔고 조회 오류: {e}")
            return 0
    
    def get_order(self, uuid):
        try:
            if self.mode == "live":
                if self.upbit is None:
                    logger.error("업비트 연결이 없습니다")
                    return None
                    
                return self.upbit.get_order(uuid)
            else:
                return {'state': 'done'}
        except Exception as e:
            logger.error(f"주문 조회 오류: {e}")
            return None
    
    def get_open_orders(self, ticker=None):
        try:
            if self.mode == "live":
                if self.upbit is None:
                    logger.error("업비트 연결이 없습니다")
                    return []
                    
                return self.upbit.get_order(ticker)
            else:
                return []
        except Exception as e:
            logger.error(f"미체결 주문 조회 오류: {e}")
            return []
    
    def cancel_order(self, uuid):
        try:
            if self.mode == "live":
                if self.upbit is None:
                    logger.error("업비트 연결이 없습니다")
                    return None
                    
                return self.upbit.cancel_order(uuid)
            else:
                return {'status': 'success'}
        except Exception as e:
            logger.error(f"주문 취소 오류: {e}")
            return None
    
    def get_avg_buy_price(self, ticker):
        try:
            if self.mode == "live":
                if self.upbit is None:
                    logger.error("업비트 연결이 없습니다")
                    return 0
                    
                return self.upbit.get_avg_buy_price(ticker)
            else:
                return self.paper_account.get_avg_buy_price(ticker)
        except Exception as e:
            logger.error(f"평균 매수가 조회 오류: {e}")
            return 0
    
    def get_current_price(self, ticker):
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"{ticker} 현재가 조회 시도 {attempt+1}/{max_retries}")
                raw_price_data = pyupbit.get_current_price(ticker)
                
                if raw_price_data is None:
                    logger.warning(f"{ticker} 현재가가 None으로 반환됨 (시도 {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                
                if isinstance(raw_price_data, dict):
                    current_price = raw_price_data.get('trade_price') or raw_price_data.get('trade_value')
                else:
                    current_price = raw_price_data
                
                if current_price:
                    logger.debug(f"{ticker} 최종 가격 값: {current_price}")
                    return current_price
                    
                logger.warning(f"{ticker} 현재가를 찾을 수 없습니다: {raw_price_data}")
                time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"{ticker} 현재가 조회 오류 (시도 {attempt+1}/{max_retries}): {e}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error(f"{ticker} 현재가 조회 최종 실패 ({max_retries}회 시도)")
        return None
    
    def buy_limit_order(self, ticker, price, quantity):
        try:
            if self.mode == "live":
                if self.upbit is None:
                    logger.error("업비트 연결이 없습니다")
                    return None
                    
                order = self.upbit.buy_limit_order(ticker, price, quantity)
                logger.info(f"실제 지정가 매수 주문: {ticker}, 가격: {price}, 수량: {quantity}")
                return order
            else:
                current_price = self.get_current_price(ticker)
                
                if current_price and current_price <= price:
                    return self.paper_account.buy(ticker, current_price, quantity)
                else:
                    order_id = f"limit_buy_{ticker}_{int(time.time())}"
                    self.paper_account.register_limit_order('buy', ticker, price, quantity, order_id)
                    logger.info(f"가상 지정가 매수 주문 등록: {ticker}, 가격: {price}, 수량: {quantity}")
                    return {'uuid': order_id, 'ticker': ticker, 'price': price, 'qty': quantity, 'type': 'limit_buy'}
        except Exception as e:
            logger.error(f"지정가 매수 주문 오류: {e}")
            return None
    
    def sell_limit_order(self, ticker, price, quantity):
        try:
            if self.mode == "live":
                if self.upbit is None:
                    logger.error("업비트 연결이 없습니다")
                    return None
                    
                order = self.upbit.sell_limit_order(ticker, price, quantity)
                logger.info(f"실제 지정가 매도 주문: {ticker}, 가격: {price}, 수량: {quantity}")
                return order
            else:
                current_price = self.get_current_price(ticker)
                
                if current_price and current_price >= price:
                    return self.paper_account.sell(ticker, current_price, quantity)
                else:
                    order_id = f"limit_sell_{ticker}_{int(time.time())}"
                    self.paper_account.register_limit_order('sell', ticker, price, quantity, order_id)
                    logger.info(f"가상 지정가 매도 주문 등록: {ticker}, 가격: {price}, 수량: {quantity}")
                    return {'uuid': order_id, 'ticker': ticker, 'price': price, 'qty': quantity, 'type': 'limit_sell'}
        except Exception as e:
            logger.error(f"지정가 매도 주문 오류: {e}")
            return None
    
    def update_limit_orders(self):
        if self.mode == "live" or not hasattr(self.paper_account, 'limit_orders'):
            return
            
        for order_id, order in list(self.paper_account.limit_orders.items()):
            current_price = self.get_current_price(order['ticker'])
            
            if current_price is None:
                continue
                
            if order['type'] == 'buy' and current_price <= order['price']:
                self.paper_account.buy(order['ticker'], order['price'], order['quantity'])
                del self.paper_account.limit_orders[order_id]
                logger.info(f"가상 지정가 매수 주문 체결: {order['ticker']}, 가격: {order['price']}, 수량: {order['quantity']}")
                
            elif order['type'] == 'sell' and current_price >= order['price']:
                self.paper_account.sell(order['ticker'], order['price'], order['quantity'])
                del self.paper_account.limit_orders[order_id]
                logger.info(f"가상 지정가 매도 주문 체결: {order['ticker']}, 가격: {order['price']}, 수량: {order['quantity']}")
'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ {file_path} 수정 완료")

def update_risk_manager():
    """risk_manager.py 수정"""
    file_path = "trading/risk_management.py"
    create_backup(file_path)
    
    content = '''from datetime import datetime, timedelta
from config.settings import STOP_LOSS_THRESHOLD, TAKE_PROFIT_THRESHOLD, TRAILING_STOP_ACTIVATION, TRAILING_STOP_DISTANCE
from utils.logger import setup_logger
from data.collector import UpbitDataCollector

logger = setup_logger("risk_manager")

class RiskManager:
    """리스크 관리 모듈"""
    
    def __init__(self):
        self.stop_levels = {}
        self.max_prices = {}
        self.collector = UpbitDataCollector()
        
    def get_volatility(self, ticker):
        """코인 변동성 계산"""
        try:
            df = self.collector.get_ohlcv(ticker, interval="day", count=20)
            if df is not None:
                return df['close'].pct_change().std() * np.sqrt(252)
            return 0.01
        except:
            return 0.01
    
    def check_trade(self, ticker, price, position, position_info, signal, account_info=None):
        result = {
            'allow_trade': True,
            'reason': None,
            'position_size': 0.3,
            'stop_loss': None,
            'take_profit': None,
            'trailing_stop': None
        }
        
        if price is None:
            result['allow_trade'] = False
            result['reason'] = 'no_price'
            return result
        
        volatility = self.get_volatility(ticker)
        stop_loss_threshold = STOP_LOSS_THRESHOLD * (1 + volatility)
        take_profit_threshold = TAKE_PROFIT_THRESHOLD * (1 + volatility)
        
        if position == 0 and signal == 1:
            if account_info:
                balance = account_info.get_balance()
                total_value = account_info.get_portfolio_value({ticker: price})
                
                if total_value > 0:
                    invested_ratio = 1 - (balance / total_value)
                    if invested_ratio > 0.5:
                        result['position_size'] = max(0.1, 0.3 - (invested_ratio - 0.5))
                        logger.debug(f"자산 분산을 위해 포지션 크기 조정: {result['position_size']:.2f}")
                
                current_hour = datetime.now().hour
                if 0 <= current_hour < 5:
                    result['position_size'] *= 0.7
                    logger.debug(f"변동성이 높은 시간대로 포지션 크기 감소: {result['position_size']:.2f}")
        
        elif position == 1:
            if not position_info:
                position_info = {'entry_price': price, 'entry_time': datetime.now() - timedelta(days=1)}
            
            entry_price = position_info.get('entry_price', price)
            entry_time = position_info.get('entry_time', datetime.now() - timedelta(days=1))
            stop_loss_price = entry_price * (1 - stop_loss_threshold)
            take_profit_price = entry_price * (1 + take_profit_threshold)
            
            current_profit_percent = (price / entry_price) - 1
            if ticker not in self.max_prices or price > self.max_prices[ticker]:
                self.max_prices[ticker] = price
            
            max_price = self.max_prices.get(ticker, price)
            
            if current_profit_percent >= TRAILING_STOP_ACTIVATION:
                trailing_stop_price = max_price * (1 - TRAILING_STOP_DISTANCE)
                if trailing_stop_price > stop_loss_price:
                    stop_loss_price = trailing_stop_price
                    logger.debug(f"{ticker} 추적 손절매 설정: {stop_loss_price:,.0f}원 (고점 대비 {TRAILING_STOP_DISTANCE*100:.1f}%)")
            
            if signal == -1:
                result['allow_trade'] = True
            elif price <= stop_loss_price:
                result['allow_trade'] = True
                result['reason'] = 'stop_loss'
            elif price >= take_profit_price:
                result['allow_trade'] = True
                result['reason'] = 'take_profit'
            else:
                result['allow_trade'] = False
                result['reason'] = 'hold_position'
            
            result['stop_loss'] = stop_loss_price
            result['take_profit'] = take_profit_price
            
            holding_time = datetime.now() - entry_time
            if holding_time.total_seconds() < 3600 and signal == -1:
                if price > entry_price * 0.97:
                    result['allow_trade'] = False
                    result['reason'] = 'minimum_holding_time'
                    logger.debug(f"{ticker} 최소 보유 시간 미달: {holding_time.seconds//60}분 (최소 60분)")
        
        if account_info and result['allow_trade'] and position == 0 and signal == 1:
            holdings_count = sum(1 for amount in account_info.holdings.values() if amount > 0)
            balance = account_info.get_balance()
            total_value = account_info.get_portfolio_value({ticker: price})
            
            if balance / total_value < 0.2 and holdings_count >= 3:
                result['allow_trade'] = False
                result['reason'] = 'cash_reserve_low'
                logger.debug(f"현금 보유량 부족: {balance/total_value:.1%} (최소 20%)")
        
        return result
    
    def set_manual_stop_loss(self, ticker, price):
        if ticker not in self.stop_levels:
            self.stop_levels[ticker] = {}
            
        self.stop_levels[ticker]['stop_loss'] = price
        logger.info(f"{ticker} 수동 손절가 설정: {price:,.0f}원")
        
    def set_manual_take_profit(self, ticker, price):
        if ticker not in self.stop_levels:
            self.stop_levels[ticker] = {}
            
        self.stop_levels[ticker]['take_profit'] = price
        logger.info(f"{ticker} 수동 익절가 설정: {price:,.0f}원")
        
    def reset_trailing_stop(self, ticker):
        if ticker in self.max_prices:
            del self.max_prices[ticker]
        
        if ticker in self.stop_levels and 'trailing_stop' in self.stop_levels[ticker]:
            del self.stop_levels[ticker]['trailing_stop']
            
        logger.info(f"{ticker} 추적 손절매 리셋")
    
    def calculate_position_size(self, balance, ticker, price, volatility=None):
        position_size = 0.3
        
        if balance > 10000000:
            position_size = 0.2
        elif balance < 500000:
            position_size = 0.5
        
        if volatility:
            if volatility > 0.05:
                position_size *= 0.7
            elif volatility < 0.02:
                position_size *= 1.2
        
        position_size = max(0.1, min(0.5, position_size))
        return position_size
    
    def calculate_kelly_criterion(self, win_rate, win_loss_ratio):
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        if kelly <= 0:
            return 0
        kelly_half = kelly * 0.5
        return min(0.5, kelly_half)
    
    def update_strategy(self, ticker, price, profit_loss_data=None):
        if not profit_loss_data:
            return
            
        consecutive_losses = profit_loss_data.get('consecutive_losses', 0)
        
        if consecutive_losses >= 3:
            logger.warning(f"{ticker} 연속 {consecutive_losses}회 손실, 리스크 감소 적용")
            
            self.stop_levels[ticker] = self.stop_levels.get(ticker, {})
            self.stop_levels[ticker]['custom_stop_loss_ratio'] = STOP_LOSS_THRESHOLD * 0.7
            self.stop_levels[ticker]['custom_take_profit_ratio'] = TAKE_PROFIT_THRESHOLD * 0.7
            return True
            
        return False
'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ {file_path} 수정 완료")

def update_validate_setup():
    """validate_setup.py 수정"""
    file_path = "validate_setup.py"
    create_backup(file_path)
    
    content = '''#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pyupbit

def validate_environment():
    """환경 검증"""
    print("🔍 환경 검증 중...")
    
    if not Path('.env').exists():
        print("❌ .env 파일이 없습니다")
        return False
    
    required_vars = ['TRADING_MODE', 'LOG_LEVEL']
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ {var} 환경 변수가 설정되지 않았습니다")
            return False
    
    if os.getenv('TRADING_MODE') == 'live':
        api_key = os.getenv('UPBIT_ACCESS_KEY', '')
        secret_key = os.getenv('UPBIT_SECRET_KEY', '')
        if not api_key or api_key == 'your_access_key_here':
            print("❌ 실제 거래를 위해서는 UPBIT_ACCESS_KEY가 필요합니다")
            return False
        try:
            upbit = pyupbit.Upbit(api_key, secret_key)
            upbit.get_balance("KRW")
        except:
            print("❌ API 키가 유효하지 않습니다")
            return False
    
    print("✅ 환경 검증 완료")
    return True

def validate_dependencies():
    """의존성 검증"""
    print("📦 의존성 검증 중...")
    
    required = ['pyupbit', 'pandas', 'numpy', 'scikit-learn', 'python-dotenv']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ 누락된 패키지: {', '.join(missing)}")
        print(f"   설치: pip install {' '.join(missing)}")
        return False
    
    print("✅ 의존성 검증 완료")
    return True

if __name__ == "__main__":
    print("🛡️ 시스템 검증 시작\n")
    
    if validate_environment() and validate_dependencies():
        print("\n🎉 모든 검증 통과!")
        print("   python main.py --mode paper 로 시작하세요")
        sys.exit(0)
    else:
        print("\n❌ 검증 실패. 문제를 해결하고 다시 실행하세요")
        sys.exit(1)
'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ {file_path} 수정 완료")

def update_sentiment_strategy():
    """sentiment_strategy.py 수정"""
    file_path = "strategy/sentiment_strategy.py"
    create_backup(file_path)
    
    content = '''import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import json

from strategy.base import BaseStrategy
from sentiment.data_collector import SentimentDataCollector
from sentiment.analyzer import SentimentAnalyzer

logger = logging.getLogger("sentiment_strategy")

class SentimentStrategy(BaseStrategy):
    """감정 분석 기반 거래 전략"""
    
    def __init__(self, name="감정 분석 전략", params=None):
        super().__init__(name)
        self.params = params or {
            'sentiment_threshold': 0.2,
            'sentiment_weight': 0.4,
            'sentiment_lookback': 3,
            'extreme_sentiment_threshold': 0.7,
            'contrarian_threshold': 0.6,
            'use_contrarian': True,
            'min_news_count': 5
        }
        
        self.sentiment_collector = SentimentDataCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_cache = {}
        self.cache_dir = 'data_cache/sentiment_strategy'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"{name} 전략 초기화 완료")
    
    def _get_recent_sentiment(self, ticker, days=3):
        """최근 감정 데이터 조회"""
        cache_file = os.path.join(self.cache_dir, f"{ticker}_sentiment.json")
        
        if os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < 3600:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    logger.debug(f"{ticker} 감정 데이터 캐시에서 로드됨")
                    return cached_data
                except:
                    pass
        
        sentiment_data = self.sentiment_collector.compile_sentiment_data(ticker)
        with open(cache_file, 'w') as f:
            json.dump(sentiment_data, f)
        
        return sentiment_data
    
    def generate_sentiment_indicators(self, ticker, df):
        result_df = df.copy()
        sentiment_data = self._get_recent_sentiment(ticker)
        
        result_df['sentiment_score'] = 0
        result_df['sentiment_signal'] = 0
        result_df['sentiment_trend'] = 0
        result_df['extreme_sentiment'] = 0
        
        if not sentiment_data:
            return result_df
            
        scores = self.sentiment_analyzer.calculate_sentiment_score(sentiment_data)
        overall_score = scores.get('overall_score', 0)
        
        if scores.get('components', {}).get('news', {}).get('count', 0) >= self.params['min_news_count']:
            result_df['sentiment_score'] = overall_score
            
            if overall_score > self.params['sentiment_threshold']:
                result_df['sentiment_signal'] = 1
            elif overall_score < -self.params['sentiment_threshold']:
                result_df['sentiment_signal'] = -1
                
            if self.params['use_contrarian']:
                if overall_score > self.params['contrarian_threshold']:
                    result_df['extreme_sentiment'] = -1
                elif overall_score < -self.params['contrarian_threshold']:
                    result_df['extreme_sentiment'] = 1
        
        return result_df
    
    def generate_signal(self, df):
        ticker = df.name if hasattr(df, 'name') else 'KRW-BTC'
        df_with_sentiment = self.generate_sentiment_indicators(ticker, df)
        
        technical_signal = self._generate_technical_signal(df)
        sentiment_signal = df_with_sentiment['sentiment_signal']
        extreme_sentiment = df_with_sentiment['extreme_sentiment']
        
        combined_signal = pd.Series(0, index=df.index)
        weight_tech = 1 - self.params['sentiment_weight']
        weight_sentiment = self.params['sentiment_weight']
        
        for i in range(len(df)):
            if extreme_sentiment.iloc[i] != 0:
                combined_signal.iloc[i] = extreme_sentiment.iloc[i]
            else:
                tech_value = technical_signal.iloc[i] if i < len(technical_signal) else 0
                sent_value = sentiment_signal.iloc[i]
                weighted_signal = (tech_value * weight_tech) + (sent_value * weight_sentiment)
                
                if weighted_signal > 0.3:
                    combined_signal.iloc[i] = 1
                elif weighted_signal < -0.3:
                    combined_signal.iloc[i] = -1
        
        return combined_signal
    
    def _generate_technical_signal(self, df):
        if not all(col in df.columns for col in ['close', 'ma5', 'ma20', 'rsi']):
            return pd.Series(0, index=df.index)
        
        signal = pd.Series(0, index=df.index)
        
        try:
            ma_cross_up = (df['ma5'].shift(1) < df['ma20'].shift(1)) & (df['ma5'] > df['ma20'])
            ma_cross_down = (df['ma5'].shift(1) > df['ma20'].shift(1)) & (df['ma5'] < df['ma20'])
            rsi_oversold = df['rsi'] < 30
            rsi_overbought = df['rsi'] > 70
            
            buy_signal = ma_cross_up | rsi_oversold
            signal[buy_signal] = 1
            
            sell_signal = ma_cross_down | rsi_overbought
            signal[sell_signal] = -1
            
        except Exception as e:
            logger.error(f"기술적 신호 생성 중 오류: {e}")
        
        return signal
'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ {file_path} 수정 완료")

def update_data_validator():
    """data_validator.py 수정"""
    file_path = "data/data_validator.py"
    create_backup(file_path)
    
    content = '''import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("data_validator")

def validate_and_clean_data(df, min_required_days=30):
    """데이터 검증 및 정제"""
    if df is None:
        return None, "data_not_available"
    
    if len(df) < min_required_days:
        logger.warning(f"데이터 부족: {len(df)} 행 (최소 {min_required_days} 필요)")
        return None, "insufficient_data"
    
    missing_count = df.isnull().sum().sum()
    missing_ratio = missing_count / (df.shape[0] * df.shape[1])
    
    if missing_ratio > 0.3:
        logger.warning(f"결측치 비율이 너무 높음: {missing_ratio:.1%}")
        return None, "too_many_missing"
    
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.interpolate(method='time').ffill().bfill()
    
    if 'volume' in cleaned_df.columns:
        volume_median = cleaned_df['volume'].median()
        cleaned_df.loc[cleaned_df['volume'] <= 0, 'volume'] = volume_median
        
    if 'close' in cleaned_df.columns:
        price_change = cleaned_df['close'].pct_change().abs()
        extreme_change = price_change > 0.5
        
        if extreme_change.any():
            extreme_dates = cleaned_df.index[extreme_change]
            logger.warning(f"급격한 가격 변동 감지: {len(extreme_dates)} 행")
            
            for date in extreme_dates:
                idx = cleaned_df.index.get_loc(date)
                if 0 < idx < len(cleaned_df) - 1:
                    prev_price = cleaned_df['close'].iloc[idx-1]
                    next_price = cleaned_df['close'].iloc[idx+1]
                    cleaned_df.loc[date, 'close'] = (prev_price + next_price) / 2
                    logger.debug(f"이상치 보정: {date}, {cleaned_df.loc[date, 'close']:.0f}원")
    
    cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
    
    quality_score = 100
    quality_score -= missing_ratio * 100
    
    if len(cleaned_df) < 60:
        quality_score -= (60 - len(cleaned_df)) / 60 * 30
    
    if 'close' in cleaned_df.columns:
        extreme_ratio = extreme_change.mean() if len(extreme_change) > 0 else 0
        quality_score -= extreme_ratio * 50
    
    logger.info(f"데이터 검증 완료: 품질 점수 {quality_score:.1f}/100, 행 수 {len(cleaned_df)}")
    
    return cleaned_df, "ok" if quality_score >= 70 else "low_quality"

def check_data_freshness(df, max_staleness_hours=24):
    if df is None or len(df) == 0:
        return False, "no_data"
    
    last_time = df.index[-1]
    time_diff = pd.Timestamp.now() - last_time
    staleness_hours = time_diff.total_seconds() / 3600
    
    if staleness_hours > max_staleness_hours:
        logger.warning(f"데이터가 오래됨: {staleness_hours:.1f}시간 ({max_staleness_hours}시간 초과)")
        return False, "data_too_old"
    
    return True, "fresh_data"

def validate_ohlcv_data(df):
    if df is None:
        return False, "data_not_available"
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"필수 컬럼 누락: {', '.join(missing_columns)}")
        return False, "missing_columns"
    
    inconsistent_rows = (
        (df['high'] < df['low']) | 
        (df['close'] > df['high']) | 
        (df['close'] < df['low']) | 
        (df['open'] > df['high']) | 
        (df['open'] < df['low'])
    )
    
    inconsistent_count = inconsistent_rows.sum()
    if inconsistent_count > 0:
        logger.warning(f"비일관적 OHLC 데이터: {inconsistent_count} 행")
        if inconsistent_count / len(df) > 0.1:
            return False, "inconsistent_data"
    
    if (df['volume'] <= 0).any():
        zero_volume_ratio = (df['volume'] <= 0).mean()
        logger.warning(f"거래량 0 또는 음수: {zero_volume_ratio:.1%} 행")
        if zero_volume_ratio > 0.2:
            return False, "invalid_volume"
    
    return True, "valid_data"

def detect_outliers(series, method='IQR'):
    """이상치 감지 (IQR 방식)"""
    if len(series) < 10:
        return pd.Series(False, index=series.index)
    
    if method == 'IQR':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
    
    rolling_mean = series.rolling(window=30, min_periods=5).mean()
    rolling_std = series.rolling(window=30, min_periods=5).std()
    rolling_mean = rolling_mean.fillna(series.mean())
    rolling_std = rolling_std.fillna(series.std())
    z_scores = (series - rolling_mean) / rolling_std
    return abs(z_scores) > 3

def fix_outliers(df, method='IQR'):
    """이상치 수정"""
    fixed_df = df.copy()
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            outliers = detect_outliers(df[col], method=method)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"{col} 컬럼에서 {outlier_count}개 이상치 감지")
                rolling_mean = df[col].rolling(window=5, min_periods=1).mean()
                fixed_df.loc[outliers, col] = rolling_mean.loc[outliers]
    
    return fixed_df
'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ {file_path} 수정 완료")

def update_visualization():
    """visualization.py 수정"""
    file_path = "visualization.py"
    create_backup(file_path)
    
    content = '''import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_technical_indicators(df, ticker, save_path=None):
    """기술적 지표 시각화"""
    print(f"{ticker} 기술적 지표 차트 생성")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['close'], label='Price')
    if 'ma5' in df.columns:
        plt.plot(df.index, df['ma5'], label='MA5')
    if 'ma20' in df.columns:
        plt.plot(df.index, df['ma20'], label='MA20')
    if 'ma60' in df.columns:
        plt.plot(df.index, df['ma60'], label='MA60')
    plt.title(f'{ticker} Price and Moving Averages')
    plt.legend()
    
    if 'rsi' in df.columns:
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['rsi'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-')
        plt.axhline(y=30, color='g', linestyle='-')
        plt.title('RSI')
        plt.legend()
    
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['macd'], label='MACD')
        plt.plot(df.index, df['macd_signal'], label='Signal')
        plt.bar(df.index, df['macd'] - df['macd_signal'], label='Histogram')
        plt.title('MACD')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"차트 저장됨: {save_path}")
    
    plt.close()
    return True

def plot_strategy_signals(df, signals, ticker, save_path=None):
    """전략 신호 시각화"""
    print(f"{ticker} 전략 신호 차트 생성")
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df.index, df['close'], label='Price')
    
    buy_signals = signals == 1
    sell_signals = signals == -1
    
    plt.scatter(df.index[buy_signals], df.loc[buy_signals, 'close'], 
                marker='^', color='g', s=100, label='Buy')
    plt.scatter(df.index[sell_signals], df.loc[sell_signals, 'close'], 
                marker='v', color='r', s=100, label='Sell')
    
    plt.title(f'{ticker} Strategy Signals')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"차트 저장됨: {save_path}")
    
    plt.close()
    return True

def plot_multi_timeframe_signals(data_dict, signals_dict, ticker, save_path=None):
    """멀티 타임프레임 신호 시각화"""
    print(f"{ticker} 멀티 타임프레임 신호 차트 생성")
    
    timeframes = list(data_dict.keys())
    n_frames = len(timeframes)
    
    plt.figure(figsize=(15, n_frames * 4))
    
    for i, tf in enumerate(timeframes):
        plt.subplot(n_frames, 1, i+1)
        
        df = data_dict[tf]
        signals = signals_dict.get(tf, pd.Series(0, index=df.index))
        
        plt.plot(df.index, df['close'], label=f'{tf} Price')
        
        buy_signals = signals == 1
        sell_signals = signals == -1
        
        if buy_signals.any():
            plt.scatter(df.index[buy_signals], df.loc[buy_signals, 'close'], 
                        marker='^', color='g', s=100, label='Buy')
        if sell_signals.any():
            plt.scatter(df.index[sell_signals], df.loc[sell_signals, 'close'], 
                        marker='v', color='r', s=100, label='Sell')
        
        plt.title(f'{ticker} - {tf} Timeframe')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"차트 저장됨: {save_path}")
    
    plt.close()
    return True

def backtest_strategy(df, signals, initial_balance=1000000, commission=0.0005):
    """백테스팅 수행"""
    balance = initial_balance
    holdings = {}
    trades = []
    
    for i in range(1, len(df)):
        if signals.iloc[i] == 1 and not holdings.get('asset'):
            amount = balance * 0.99
            quantity = amount / df['close'].iloc[i]
            balance -= amount
            holdings['asset'] = {'quantity': quantity, 'price': df['close'].iloc[i]}
            trades.append({
                'timestamp': df.index[i],
                'type': 'buy',
                'price': df['close'].iloc[i],
                'quantity': quantity
            })
        elif signals.iloc[i] == -1 and holdings.get('asset'):
            quantity = holdings['asset']['quantity']
            amount = quantity * df['close'].iloc[i] * (1 - commission)
            balance += amount
            trades.append({
                'timestamp': df.index[i],
                'type': 'sell',
                'price': df['close'].iloc[i],
                'quantity': quantity
            })
            holdings.pop('asset')
    
    final_value = balance
    if holdings.get('asset'):
        final_value += holdings['asset']['quantity'] * df['close'].iloc[-1]
    
    returns = pd.Series([t['price'] / t['price'] for t in trades if t['type'] == 'sell'], index=[t['timestamp'] for t in trades if t['type'] == 'sell'])
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    return {
        'final_value': final_value,
        'return': (final_value / initial_balance - 1) * 100,
        'sharpe_ratio': sharpe_ratio,
        'trades': trades
    }

def generate_strategy_report(df, signals, ticker, save_path=None):
    """전략 성과 보고서 생성"""
    print(f"{ticker} 전략 보고서 생성")
    
    buy_count = sum(signals == 1)
    sell_count = sum(signals == -1)
    
    backtest_results = backtest_strategy(df, signals)
    
    report = f"=== {ticker} 전략 보고서 ===\n"
    report += f"기간: {df.index[0]} ~ {df.index[-1]}\n"
    report += f"매수 신호: {buy_count}개\n"
    report += f"매도 신호: {sell_count}개\n"
    report += f"최종 가치: {backtest_results['final_value']:,.0f}원\n"
    report += f"수익률: {backtest_results['return']:.2f}%\n"
    report += f"샤프 비율: {backtest_results['sharpe_ratio']:.2f}\n"
    report += f"총 거래 횟수: {len(backtest_results['trades'])}건\n"
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"보고서 저장됨: {save_path}")
    
    print(report)
    return report
'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ {file_path} 수정 완료")

def update_env():
    """env 설정 가이드 출력"""
    print("\n=== .env 설정 가이드 ===")
    print("아래 설정을 .env 파일에 추가하거나 수정하세요:")
    print("""
# 코인별 리스크 관리 설정
STOP_LOSS_THRESHOLD_BTC=0.03
STOP_LOSS_THRESHOLD_ALT=0.05
TAKE_PROFIT_THRESHOLD_BTC=0.08
TAKE_PROFIT_THRESHOLD_ALT=0.1

# 텔레그램 알림 (선택)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
""")
    print("⚠️ API 키(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)는 안전하게 관리하세요!")
    print("======================")

def main():
    """모든 수정 수행"""
    print("🚀 시스템 업데이트 시작\n")
    
    update_ml_strategy()
    update_trading_executor()
    update_order_manager()
    update_risk_manager()
    update_validate_setup()
    update_sentiment_strategy()
    update_data_validator()
    update_visualization()
    update_env()
    
    print("\n🎉 모든 파일 수정 완료!")
    print("다음 명령으로 시스템을 실행하세요:")
    print("python main.py --mode paper --balance 1000000")
    print("\n검증 명령:")
    print("python validate_setup.py")
    print("python validate_models.py")

if __name__ == "__main__":
    from datetime import datetime
    main()

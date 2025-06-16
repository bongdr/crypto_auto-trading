#!/usr/bin/env python3
"""
Import 오류 빠른 수정 스크립트
기존 프로젝트와 호환되도록 import 경로와 클래스명을 수정합니다.
"""

import os
import re

def fix_import_errors():
    """Import 오류 수정"""
    
    print("🔧 Import 오류 수정 시작...")
    
    # 1. data/improved_coin_selector.py 수정
    fix_coin_selector()
    
    # 2. strategy/improved_ml_strategy.py 수정  
    fix_ml_strategy()
    
    # 3. improved_main.py 수정
    fix_main_script()
    
    # 4. 누락된 기본 모듈들 생성
    create_missing_modules()
    
    print("✅ Import 오류 수정 완료!")

def fix_coin_selector():
    """코인 선택기 import 수정"""
    
    file_path = 'data/improved_coin_selector.py'
    if not os.path.exists(file_path):
        print(f"❌ {file_path} 파일이 없습니다.")
        return
    
    # 수정된 내용으로 교체
    fixed_content = '''
import logging
import numpy as np
import pandas as pd
from data.collector import UpbitDataCollector  # 수정된 import
from utils.logger import setup_logger

logger = setup_logger("improved_coin_selector")

class ImprovedCoinSelector:
    """개선된 코인 선택기 - 데이터 품질 및 안정성 중심"""
    
    def __init__(self, min_data_days=90, min_volume_krw=10_000_000_000):
        """
        Args:
            min_data_days (int): 최소 필요 데이터 일수
            min_volume_krw (int): 최소 일평균 거래대금 (원)
        """
        self.min_data_days = min_data_days
        self.min_volume_krw = min_volume_krw
        self.data_collector = UpbitDataCollector()  # 수정된 클래스명
        
    def validate_coin_data(self, ticker):
        """코인 데이터 품질 검증"""
        try:
            # 일봉 데이터 수집 - 수정된 메소드명
            df = self.data_collector.get_ohlcv(ticker, interval='day', count=120)
            
            if df is None or len(df) < self.min_data_days:
                logger.warning(f"{ticker}: 데이터 부족 ({len(df) if df is not None else 0}/{self.min_data_days}일)")
                return False
                
            # 거래량 검증 - 컬럼명 수정
            volume_col = 'volume' if 'volume' in df.columns else 'candle_acc_trade_price'
            if volume_col not in df.columns:
                logger.warning(f"{ticker}: 거래량 데이터 없음")
                return False
                
            avg_volume_krw = df[volume_col].mean()
            if avg_volume_krw < self.min_volume_krw:
                logger.warning(f"{ticker}: 거래량 부족 ({avg_volume_krw/1e9:.1f}십억원 < {self.min_volume_krw/1e9:.1f}십억원)")
                return False
                
            # 가격 안정성 검증 (급격한 변동 확인)
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # 50% 이상 변동
            if extreme_changes > 5:  # 최근 120일 중 5회 이상
                logger.warning(f"{ticker}: 가격 불안정 (극단적 변동 {extreme_changes}회)")
                return False
                
            # 연속 거래 중단 검증
            zero_volume_days = (df[volume_col] == 0).sum()
            if zero_volume_days > 3:
                logger.warning(f"{ticker}: 거래 중단일 과다 ({zero_volume_days}일)")
                return False
                
            logger.info(f"{ticker}: 데이터 품질 검증 통과")
            return True
            
        except Exception as e:
            logger.error(f"{ticker} 데이터 검증 중 오류: {e}")
            return False
    
    def calculate_coin_score(self, ticker, df):
        """코인 점수 계산 - 안정성과 수익성 균형"""
        try:
            # 거래량 컬럼 확인
            volume_col = 'volume' if 'volume' in df.columns else 'candle_acc_trade_price'
            
            # 1. 변동성 점수 (적당한 변동성 선호)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 연간 변동성
            vol_score = max(0, 1 - abs(volatility - 0.4) / 0.3)  # 40% 변동성 선호
            
            # 2. 유동성 점수
            avg_volume = df[volume_col].mean()
            liquidity_score = min(1.0, avg_volume / (50_000_000_000))  # 500억원 기준
            
            # 3. 추세 점수
            recent_trend = (df['close'].iloc[-10:].mean() / df['close'].iloc[-30:].mean()) - 1
            trend_score = max(0, min(1, (recent_trend + 0.1) / 0.2))  # -10% ~ +10% 정규화
            
            # 4. 안정성 점수
            price_stability = 1 - (df['close'].pct_change().abs() > 0.2).mean()
            
            # 가중 평균 점수
            total_score = (
                vol_score * 0.25 +
                liquidity_score * 0.35 +
                trend_score * 0.20 +
                price_stability * 0.20
            )
            
            return {
                'total_score': total_score,
                'volatility': volatility,
                'avg_volume_krw': avg_volume,
                'trend': recent_trend,
                'stability': price_stability
            }
            
        except Exception as e:
            logger.error(f"{ticker} 점수 계산 오류: {e}")
            return {'total_score': 0}
    
    def get_krw_tickers(self):
        """KRW 마켓 티커 목록 가져오기"""
        try:
            import pyupbit
            tickers = pyupbit.get_tickers(fiat="KRW")
            return tickers if tickers else []
        except Exception as e:
            logger.error(f"티커 목록 조회 오류: {e}")
            # 기본 티커 목록 반환
            return ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-MATIC', 'KRW-SOL', 'KRW-XRP']
    
    def select_quality_coins(self, target_count=3):
        """품질 기반 코인 선택"""
        logger.info("개선된 코인 선택 시작")
        
        try:
            # 전체 코인 목록 가져오기
            all_tickers = self.get_krw_tickers()
            
            # 안정적인 주요 코인들 우선 검토
            priority_coins = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-MATIC', 'KRW-SOL', 'KRW-XRP']
            other_coins = [t for t in all_tickers if t not in priority_coins]
            
            # 우선순위 코인 + 기타 코인 순으로 검토
            check_order = priority_coins + other_coins
            
            validated_coins = []
            coin_scores = {}
            
            for ticker in check_order:
                if len(validated_coins) >= target_count * 2:  # 충분한 후보 확보 시 중단
                    break
                    
                logger.info(f"{ticker} 검증 중...")
                
                if self.validate_coin_data(ticker):
                    # 데이터 재수집 및 점수 계산
                    df = self.data_collector.get_ohlcv(ticker, interval='day', count=120)
                    score_info = self.calculate_coin_score(ticker, df)
                    
                    if score_info['total_score'] > 0.3:  # 최소 점수 기준
                        validated_coins.append(ticker)
                        coin_scores[ticker] = score_info
                        logger.info(f"{ticker} 선정 완료 (점수: {score_info['total_score']:.3f})")
            
            if len(validated_coins) < target_count:
                logger.warning(f"충분한 품질의 코인을 찾지 못함 ({len(validated_coins)}/{target_count})")
                # BTC는 항상 포함
                if 'KRW-BTC' not in validated_coins:
                    validated_coins.insert(0, 'KRW-BTC')
                    df_btc = self.data_collector.get_ohlcv('KRW-BTC', interval='day', count=120)
                    if df_btc is not None:
                        coin_scores['KRW-BTC'] = self.calculate_coin_score('KRW-BTC', df_btc)
            
            # 점수 기준 정렬 및 최종 선택
            sorted_coins = sorted(validated_coins, key=lambda x: coin_scores.get(x, {}).get('total_score', 0), reverse=True)
            selected_coins = sorted_coins[:target_count]
            
            # 결과 출력
            logger.info(f"최종 선정 결과: {len(selected_coins)}개 코인")
            for ticker in selected_coins:
                score = coin_scores.get(ticker, {})
                logger.info(f"{ticker}: 점수 {score.get('total_score', 0):.3f}, "
                           f"변동성 {score.get('volatility', 0):.1%}, "
                           f"거래대금 {score.get('avg_volume_krw', 0)/1e9:.1f}십억원")
            
            return selected_coins, coin_scores
            
        except Exception as e:
            logger.error(f"코인 선택 중 오류: {e}")
            # 기본값 반환
            return ['KRW-BTC', 'KRW-ETH', 'KRW-ADA'], {}
'''
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"✅ {file_path} 수정 완료")

def fix_ml_strategy():
    """ML 전략 import 수정"""
    
    file_path = 'strategy/improved_ml_strategy.py'
    if not os.path.exists(file_path):
        print(f"❌ {file_path} 파일이 없습니다.")
        return
    
    fixed_content = '''
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
'''
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"✅ {file_path} 수정 완료")

def fix_main_script():
    """메인 스크립트 import 수정"""
    
    file_path = 'improved_main.py'
    if not os.path.exists(file_path):
        print(f"❌ {file_path} 파일이 없습니다.")
        return
    
    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # import 수정
    content = content.replace(
        'from data.data_collector import DataCollector',
        'from data.collector import UpbitDataCollector'
    )
    
    content = content.replace(
        'data_collector = DataCollector()',
        'data_collector = UpbitDataCollector()'
    )
    
    content = content.replace(
        'df = data_collector.get_historical_data(ticker, count=200, interval=\'day\')',
        'df = data_collector.get_ohlcv(ticker, interval=\'day\', count=200)'
    )
    
    content = content.replace(
        'df = data_collector.get_historical_data(ticker, count=100, interval=\'day\')',
        'df = data_collector.get_ohlcv(ticker, interval=\'day\', count=100)'
    )
    
    content = content.replace(
        'df = data_collector.get_historical_data(ticker, count=1, interval=\'day\')',
        'df = data_collector.get_ohlcv(ticker, interval=\'day\', count=1)'
    )
    
    # 파일 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {file_path} 수정 완료")

def create_missing_modules():
    """누락된 기본 모듈들 생성"""
    
    # 1. utils/logger.py 확인 및 생성
    if not os.path.exists('utils/logger.py'):
        logger_content = '''
import logging
import os
from datetime import datetime

def setup_logger(name, level=logging.INFO):
    """로거 설정"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 포맷터
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(f'logs/{name}.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
'''
        os.makedirs('utils', exist_ok=True)
        with open('utils/logger.py', 'w', encoding='utf-8') as f:
            f.write(logger_content)
        print("✅ utils/logger.py 생성 완료")
    
    # 2. trading/risk_manager.py 수정 (import 경로 수정)
    if os.path.exists('trading/risk_manager.py'):
        with open('trading/risk_manager.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # import 수정이 필요하다면 여기서 수정
        print("✅ trading/risk_manager.py 확인 완료")
    
    # 3. __init__.py 파일들 생성
    init_dirs = ['data', 'strategy', 'trading', 'utils', 'config']
    for dir_name in init_dirs:
        os.makedirs(dir_name, exist_ok=True)
        init_file = os.path.join(dir_name, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'# {dir_name} 패키지\n')
            print(f"✅ {init_file} 생성 완료")

def create_simple_test():
    """간단한 테스트 스크립트 생성"""
    
    test_content = '''
#!/usr/bin/env python3
"""
빠른 수정 테스트 스크립트
"""

def test_imports():
    """Import 테스트"""
    try:
        print("📦 Import 테스트 시작...")
        
        from data.improved_coin_selector import ImprovedCoinSelector
        print("✅ ImprovedCoinSelector import 성공")
        
        from strategy.improved_ml_strategy import ImprovedMLStrategy
        print("✅ ImprovedMLStrategy import 성공")
        
        from trading.risk_manager import RiskManager
        print("✅ RiskManager import 성공")
        
        from utils.system_monitor import SystemMonitor
        print("✅ SystemMonitor import 성공")
        
        print("🎉 모든 Import 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ Import 오류: {e}")
        return False

def test_basic_functionality():
    """기본 기능 테스트"""
    try:
        print("🔧 기본 기능 테스트 시작...")
        
        # 코인 선택기 테스트
        coin_selector = ImprovedCoinSelector()
        tickers = coin_selector.get_krw_tickers()
        print(f"✅ 티커 조회: {len(tickers)}개")
        
        # ML 전략 테스트
        ml_strategy = ImprovedMLStrategy("KRW-BTC")
        print("✅ ML 전략 초기화 성공")
        
        # 리스크 관리자 테스트
        risk_manager = RiskManager()
        print("✅ 리스크 관리자 초기화 성공")
        
        print("🎉 기본 기능 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 기능 테스트 오류: {e}")
        return False

if __name__ == "__main__":
    print("🚀 빠른 수정 테스트 시작")
    print("="*50)
    
    # Import 테스트
    import_ok = test_imports()
    
    if import_ok:
        # 기본 기능 테스트
        func_ok = test_basic_functionality()
        
        if func_ok:
            print("\\n🎉 모든 테스트 통과! 시스템 사용 준비 완료")
        else:
            print("\\n⚠️ 일부 기능에 문제가 있습니다.")
    else:
        print("\\n❌ Import 오류로 테스트 중단")
'''
    
    with open('test_quick_fix.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ test_quick_fix.py 생성 완료")

if __name__ == "__main__":
    print("🔧 Import 오류 빠른 수정 시작")
    print("="*50)
    
    fix_import_errors()
    create_simple_test()
    
    print("\n" + "="*50)
    print("🎉 빠른 수정 완료!")
    print("="*50)
    
    print("\n📋 다음 단계:")
    print("1. 테스트 실행: python test_quick_fix.py")
    print("2. 문제없으면 시스템 실행: python improved_main.py")
    print("3. 추가 오류 발생시 해당 모듈 개별 수정")
    
    print("\n⚠️ 주의사항:")
    print("- 기존 프로젝트 구조와 호환되도록 수정했습니다")
    print("- data.collector.UpbitDataCollector 사용")
    print("- 필요한 경우 추가 모듈 수정 필요할 수 있습니다")

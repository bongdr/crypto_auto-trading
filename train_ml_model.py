import os
from data.collector import UpbitDataCollector
from models.feature_engineering import FeatureEngineer
from strategy.ml_strategy import MLStrategy
from utils.visualization import plot_strategy_signals

# 모델 학습 설정
ticker = "KRW-BTC"
model_path = os.path.join("saved_models", f"{ticker}_ml_model.joblib")

# 학습 데이터 수집 (더 긴 기간의 데이터 사용)
collector = UpbitDataCollector()
train_df = collector.get_historical_data(ticker, days=120, interval="day")
print(f"학습 데이터 수집 완료: {len(train_df)} 행")

# 특성 엔지니어링
fe = FeatureEngineer()
train_df_features = fe.add_ml_features(train_df)
print(f"특성 추가 완료: {len(train_df_features.columns)} 개 특성")

# 머신러닝 전략 초기화 및 학습
ml_strategy = MLStrategy("BTC 머신러닝 전략")
ml_strategy.train_model(
    train_df_features, 
    horizon=3,  # 3일 후 가격 예측
    threshold=0.01,  # 1% 이상 상승 예측 시 매수
    test_size=0.2,
    save_path=model_path
)

# 학습된 모델로 신호 생성
ml_signals = ml_strategy.generate_signal(train_df_features)
buy_count = sum(ml_signals == 1)
sell_count = sum(ml_signals == -1)
print(f"ML 전략 신호 생성: 매수 {buy_count}, 매도 {sell_count}")

# 신호 시각화
plot_strategy_signals(
    train_df_features, 
    ml_signals, 
    ticker, 
    save_path="backtest_results/ml_strategy_signals.png"
)

# 테스트 데이터로 검증
test_df = collector.get_ohlcv(ticker, interval="day", count=60)  # 데이터 개수를 60개로 증가
if test_df is not None and len(test_df) >= 60:  # 충분한 데이터 확인
    test_df_features = fe.add_ml_features(test_df)
    if test_df_features is not None:
        test_signals = ml_strategy.generate_signal(test_df_features)
        print(f"테스트 신호 생성: {sum(test_signals != 0)} 개")
        
        # 보고서 생성 부분 수정
        try:
            from utils.visualization import generate_strategy_report
            test_report = generate_strategy_report(
                test_df_features, 
                test_signals, 
                ticker=ticker, 
                save_path="backtest_results/ml_test_report.txt"
            )
        except Exception as e:
            print(f"테스트 보고서 생성 오류: {e}")
    else:
        print("테스트 데이터에 대한 특성 생성 실패")
else:
    print("테스트 데이터가 충분하지 않거나 로드 실패")
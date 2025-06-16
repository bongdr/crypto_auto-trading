from strategy.technical import TechnicalStrategy
from strategy.ml_strategy import MLStrategy
from strategy.ensemble_strategy import EnsembleStrategy
from trading.execution import TradingExecutor
from data.coin_selector import CoinSelector
import time
import os

# 거래할 코인 선택
coin_selector = CoinSelector()
selected_coins = coin_selector.select_balanced_portfolio(top_n=3)
print(f"선택된 코인: {', '.join(selected_coins)}")

# 전략 설정
# 1. 기술적 전략
tech_strategy = TechnicalStrategy("기술적 전략")

# 2. ML 전략 (미리 학습된 모델 필요)
# ml_strategy = MLStrategy(
#     "ML 전략", 
#     model_path=os.path.join("saved_models", "KRW-BTC_ml_model.joblib")
# )

# 3. 앙상블 전략
ensemble = EnsembleStrategy("앙상블 전략")
ensemble.add_strategy(tech_strategy, weight=1.0)
# ensemble.add_strategy(ml_strategy, weight=0.8)  # ML 전략 사용 시 주석 해제

# 거래 실행기 초기화
executor = TradingExecutor(ensemble, initial_balance=1000000)

# 거래 시작
executor.start_trading(selected_coins)

try:
    # 거래 루프 실행 (백그라운드 스레드)
    print("거래 시작 (Ctrl+C로 중지)")
    executor.run_trading_loop(interval=60)  # 60초마다 업데이트
    
except KeyboardInterrupt:
    print("사용자에 의한 중지")
    executor.stop_trading()

except Exception as e:
    print(f"오류 발생: {e}")
    executor.stop_trading()
from strategy.technical import TechnicalStrategy
from strategy.ml_strategy import MLStrategy
from strategy.ensemble_strategy import EnsembleStrategy
from strategy.adaptive_ensemble import AdaptiveEnsemble
from models.market_state_detector import MarketStateDetector
from models.ml_model_manager import MLModelManager
from models.hyperparameter_optimizer import HyperparameterOptimizer
from trading.execution import TradingExecutor
from data.coin_selector import CoinSelector
from data.collector import UpbitDataCollector  # 추가
from models.feature_engineering import FeatureEngineer  # 추가
from utils.logger import setup_logger
import os
import time
from datetime import datetime
from threading import Thread

logger = setup_logger("advanced_trading")

class AdvancedTradingSystem:
    """고급 자동 거래 시스템 - 개선된 버전"""
    
    def __init__(self, initial_balance=1000000, telegram_config=None, 
             enable_sentiment=False, enable_fund_manager=False):
        """초기화"""
        self.initial_balance = initial_balance
        
        # 이제 정상적으로 사용 가능
        self.sentiment_enabled = enable_sentiment
        self.fund_manager_enabled = enable_fund_manager
        self.initial_balance = initial_balance
        
        # 코인 선택
        self.coin_selector = CoinSelector()
        self.tickers = []
        
        # 전략
        self.strategies = {}  # {ticker: {state_id: strategy}}
        self.active_strategies = {}  # {ticker: current_strategy}
        
        # 상태 감지기
        self.state_detectors = {}  # {ticker: detector}
        
        # ML 모델 관리자
        self.ml_managers = {}  # {ticker: manager}
        
        # 하이퍼파라미터 최적화
        self.optimizers = {}  # {ticker: optimizer}
        
        # 거래 실행기
        self.executor = None
        
        # 시스템 설정
        self.detection_method = 'clustering'  # 상태 감지 방법
        self.trading_interval = 600  # 거래 주기 (초)
        self.retraining_interval = 7  # 재학습 주기 (일)
        self.optimization_interval = 14  # 최적화 주기 (일)
        
        # 자동 실행
        self.running = False
        self.management_threads = {}  # 관리 스레드
        
        # 마지막 작업 시간
        self.last_retraining = {}  # {ticker: datetime}
        self.last_optimization = {}  # {ticker: datetime}

        # 텔레그램 알림 설정
        self.telegram_config = telegram_config
        self.telegram_notifier = None

        if telegram_config and telegram_config.get('enabled', False):
            try:
                from telegram_notifier import TelegramNotifier
                token = telegram_config.get('token')
                chat_id = telegram_config.get('chat_id')
                
                if token and chat_id:
                    self.telegram_notifier = TelegramNotifier(token, chat_id)
                    self.telegram_notifier.send_message("🚀 *가상화폐 자동 거래 시스템 시작*\n"
                                                    f"초기 잔고: {initial_balance:,}원")
                    
                    # 일일 보고서 예약
                    report_time = telegram_config.get('report_time', '21:00')
                    self.telegram_notifier.schedule_daily_report(report_time)
                    
                    logger.info(f"텔레그램 알림 시스템 초기화 완료")
                else:
                    logger.warning("텔레그램 토큰 또는 채팅 ID가 없습니다")
            except Exception as e:
                logger.error(f"텔레그램 알림 시스템 초기화 실패: {e}")

        # 감정 분석 모듈 초기화
        self.sentiment_enabled = enable_sentiment
        self.sentiment_collectors = {}  # {ticker: collector}
        self.sentiment_analyzer = None
        
        if enable_sentiment:
            try:
                from sentiment.data_collector import SentimentDataCollector
                from sentiment.analyzer import SentimentAnalyzer
                from strategy.sentiment_strategy import SentimentStrategy
                
                # 감정 분석 모듈 초기화
                self.sentiment_analyzer = SentimentAnalyzer()
                
                logger.info("감정 분석 모듈 초기화 완료")
            except Exception as e:
                logger.error(f"감정 분석 모듈 초기화 실패: {e}")
                self.sentiment_enabled = False
        
        # 자금 관리 모듈 초기화
        self.fund_manager_enabled = enable_fund_manager
        self.fund_manager = None
        self.portfolio_rebalancer = None
        
        if enable_fund_manager:
            try:
                from finance.fund_manager import FundManager
                from finance.portfolio_rebalancer import PortfolioRebalancer
                
                # 자금 관리자 초기화
                self.fund_manager = FundManager(initial_balance)
                
                # 포트폴리오 리밸런서 초기화
                self.portfolio_rebalancer = PortfolioRebalancer()
                
                logger.info("자금 관리 모듈 초기화 완료")
            except Exception as e:
                logger.error(f"자금 관리 모듈 초기화 실패: {e}")
                self.fund_manager_enabled = False
    
    def select_coins(self, method='balanced', count=3):
        """거래할 코인 선택"""
        if method == 'balanced':
            self.tickers = self.coin_selector.select_balanced_portfolio(top_n=count)
        elif method == 'uncorrelated':
            self.tickers = self.coin_selector.select_uncorrelated_coins(top_n=count)
        else:
            self.tickers = self.coin_selector.select_coins(top_n=count)
            
        logger.info(f"선택된 코인: {', '.join(self.tickers)}")
        return self.tickers
    
    def initialize_system(self):
        """시스템 초기화 - 개선된 버전"""
        if not self.tickers:
            self.select_coins()
            
        # 초기 데이터 수집 및 준비
        data_collector = UpbitDataCollector()
        feature_engineer = FeatureEngineer()
        
        # 데이터 준비 성공 여부 추적
        data_ready = {}
        
        for ticker in self.tickers:
            logger.info(f"{ticker} 초기 데이터 준비 시작...")
            data_ready[ticker] = False
            
            try:
                # 더 안정적인 데이터 수집
                df = None
                
                # 1차 시도: 일봉 60일
                try:
                    df = data_collector.get_historical_data(ticker, days=60, interval="day")
                    if df is not None and len(df) >= 30:
                        logger.info(f"{ticker} 일봉 데이터 수집 성공: {len(df)}행")
                except Exception as e:
                    logger.warning(f"{ticker} 일봉 데이터 수집 실패: {e}")
                
                # 2차 시도: 4시간봉
                if df is None or len(df) < 30:
                    try:
                        df = data_collector.get_ohlcv(ticker, interval="minute240", count=100)
                        if df is not None and len(df) >= 30:
                            logger.info(f"{ticker} 4시간봉 데이터로 대체: {len(df)}행")
                    except Exception as e:
                        logger.warning(f"{ticker} 4시간봉 데이터 수집 실패: {e}")
                
                # 3차 시도: 1시간봉
                if df is None or len(df) < 30:
                    try:
                        df = data_collector.get_ohlcv(ticker, interval="minute60", count=100)
                        if df is not None and len(df) >= 30:
                            logger.info(f"{ticker} 1시간봉 데이터로 대체: {len(df)}행")
                    except Exception as e:
                        logger.warning(f"{ticker} 1시간봉 데이터 수집 실패: {e}")
                
                # 최종 검증
                if df is not None and len(df) >= 30:
                    # 특성 추가 시도
                    try:
                        df = feature_engineer.add_indicators(df)
                        if df is not None:
                            logger.info(f"{ticker} 초기 데이터 준비 완료: {len(df)} 행, {len(df.columns)} 열")
                            data_ready[ticker] = True
                    except Exception as e:
                        logger.error(f"{ticker} 특성 추가 실패: {e}")
                else:
                    logger.warning(f"{ticker} 충분한 데이터를 가져오지 못했습니다.")
                    
            except Exception as e:
                logger.error(f"{ticker} 초기 데이터 준비 중 예외 발생: {e}")
        
        # 데이터가 준비된 코인만 사용
        prepared_tickers = [ticker for ticker, ready in data_ready.items() if ready]
        if not prepared_tickers:
            logger.error("데이터가 준비된 코인이 없습니다. 기본 BTC 사용")
            prepared_tickers = ['KRW-BTC']
        
        self.tickers = prepared_tickers
        logger.info(f"초기화할 코인: {', '.join(self.tickers)} ({len(self.tickers)}개)")
        
        # 여기서부터 기존 코드 계속...
        for ticker in self.tickers:
            # 상태 감지기 초기화
            self.state_detectors[ticker] = MarketStateDetector(
                ticker=ticker, 
                detection_method=self.detection_method,
                n_states=5
            )
            
            # ML 모델 관리자 초기화
            self.ml_managers[ticker] = MLModelManager([ticker])
            
            # 하이퍼파라미터 최적화 초기화 (기술적 전략)
            self.optimizers[ticker] = HyperparameterOptimizer(
                TechnicalStrategy,
                ticker=ticker,
                optimization_method='genetic'
            )
            
            # 마지막 작업 시간 초기화
            self.last_retraining[ticker] = datetime.now()
            self.last_optimization[ticker] = datetime.now()
            
            # 기본 전략 생성
            tech_strategy = TechnicalStrategy(f"{ticker} 기술적 전략")
            
            # 최적화된 파라미터가 있으면 적용
            optimized_params = self.optimizers[ticker].load_best_params()
            if optimized_params:
                tech_strategy = TechnicalStrategy(f"{ticker} 최적화 전략", optimized_params)
            
            # 전략 저장
            self.strategies[ticker] = {'default': tech_strategy}
            self.active_strategies[ticker] = tech_strategy
        
        # 실행기 초기화
        self.executor = TradingExecutor(AdaptiveEnsemble("적응형 앙상블"), initial_balance=self.initial_balance)
        
        # 앙상블에 전략 추가
        ensemble = self.executor.strategy
        for ticker, strategy in self.active_strategies.items():
            ensemble.add_strategy(strategy)
            
        # 자동 업데이트 시작
        ensemble.start_auto_update()
        
        # 감정 분석 모듈 설정 (추가됨)
        if self.sentiment_enabled:
            self.setup_sentiment_analysis()
            
            # 감정 기반 전략을 앙상블에 추가
            for ticker in self.tickers:
                if ticker in self.strategies and 'sentiment' in self.strategies[ticker]:
                    sentiment_strategy = self.strategies[ticker]['sentiment']
                    ensemble.add_strategy(sentiment_strategy, initial_weight=0.3)
                    logger.info(f"{ticker} 감정 기반 전략 앙상블에 추가됨")
        
        # 자금 관리 모듈 설정 (추가됨)
        if self.fund_manager_enabled:
            self.setup_fund_management()
        
        # 텔레그램 알림 전송
        if hasattr(self, 'telegram_notifier') and self.telegram_notifier:
            ticker_list = ", ".join(self.tickers)
            self.telegram_notifier.send_message(
                f"✅ *시스템 초기화 완료*\n"
                f"거래 코인: {ticker_list}\n"
                f"감정 분석: {'활성화' if self.sentiment_enabled else '비활성화'}\n"
                f"자금 관리: {'활성화' if self.fund_manager_enabled else '비활성화'}"
            )
        
        logger.info("시스템 초기화 완료")
        
        return True
    
    def detect_market_states(self):
        """모든 코인의 현재 시장 상태 감지"""
        current_states = {}
        
        for ticker, detector in self.state_detectors.items():
            state = detector.detect_current_state()
            if state:
                current_states[ticker] = state.state_id
                logger.info(f"{ticker} 현재 상태: {state}")
                
        return current_states
    
    def update_strategies(self):
        """현재 시장 상태에 맞는 전략으로 업데이트"""
        # 시장 상태 감지
        market_states = self.detect_market_states()
        
        # 업데이트 여부
        updated = False
        
        # 각 코인별 최적 전략 설정
        for ticker, state_id in market_states.items():
            detector = self.state_detectors[ticker]
            
            # 이전 상태 ID 저장 (수정된 부분)
            old_state_id = None
            if ticker in self.active_strategies:
                # 현재 활성 전략에서 이전 상태 추출
                current_strategy = self.active_strategies[ticker]
                if hasattr(detector, 'current_state') and detector.current_state:
                    old_state_id = detector.current_state.state_id
                elif hasattr(detector, 'previous_state_id'):
                    old_state_id = detector.previous_state_id
            
            # 시장 상태에 최적화된 전략 정보 가져오기
            strategy_info = detector.get_optimal_strategy(state_id)
            
            # 전략 정보가 없으면 기본 전략 유지
            if not strategy_info:
                continue
                
            # 이미 해당 상태의 전략이 있는지 확인
            if state_id in self.strategies[ticker]:
                # 기존 전략 사용
                strategy = self.strategies[ticker][state_id]
            else:
                # 새 전략 생성
                if strategy_info.get('type') == 'trend_following':
                    strategy = TechnicalStrategy(f"{ticker} 추세추종", strategy_info.get('params', {}))
                elif strategy_info.get('type') == 'counter_trend':
                    strategy = TechnicalStrategy(f"{ticker} 역추세", strategy_info.get('params', {}))
                else:
                    # 기본 전략
                    strategy = self.strategies[ticker]['default']
                
                # 전략 저장
                self.strategies[ticker][state_id] = strategy
                
            # 활성 전략 업데이트
            if self.active_strategies[ticker] != strategy:
                old_strategy = self.active_strategies[ticker]
                self.active_strategies[ticker] = strategy
                
                # 전략 변경 로깅
                logger.info(f"{ticker} 전략 변경: {old_strategy.get_name()} → {strategy.get_name()}")
                
                # 시장 상태 변화 알림 (수정된 부분)
                if hasattr(self, 'telegram_notifier') and self.telegram_notifier and old_state_id:
                    old_state = detector.market_states.get(old_state_id, None)
                    new_state = detector.market_states.get(state_id, None)
                    
                    if old_state and new_state:
                        old_state_info = {
                            'state_id': old_state_id,
                            'characteristics': old_state.characteristics if hasattr(old_state, 'characteristics') else {}
                        }
                        
                        new_state_info = {
                            'state_id': state_id,
                            'characteristics': new_state.characteristics if hasattr(new_state, 'characteristics') else {}
                        }
                        
                        self.telegram_notifier.notify_state_change(ticker, old_state_info, new_state_info)
                
                updated = True
        
        # 앙상블 업데이트
        if updated and self.executor:
            # 앙상블 초기화
            ensemble = AdaptiveEnsemble("적응형 앙상블")
            
            # 각 코인의 활성 전략 추가
            for ticker, strategy in self.active_strategies.items():
                ensemble.add_strategy(strategy)
                
            # 실행기에 앙상블 설정
            self.executor.strategy = ensemble
            
            # 자동 업데이트 시작
            ensemble.start_auto_update()
            
            logger.info("앙상블 전략 업데이트됨")
        
        return updated
    
    def check_retraining_needed(self, ticker):
        """ML 모델 재학습 필요성 확인"""
        if ticker not in self.last_retraining:
            return True
            
        days_since_retraining = (datetime.now() - self.last_retraining[ticker]).days
        
        if days_since_retraining >= self.retraining_interval:
            logger.info(f"{ticker} 모델 재학습 필요 (마지막 학습 후 {days_since_retraining}일 경과)")
            return True
            
        # 기타 필요 조건 확인 (성능 저하 등)
        manager = self.ml_managers.get(ticker)
        if manager and any(manager.check_retraining_needed(t) for t in [ticker]):
            return True
            
        return False
    
    def check_optimization_needed(self, ticker):
        """전략 최적화 필요성 확인"""
        if ticker not in self.last_optimization:
            return True
            
        days_since_optimization = (datetime.now() - self.last_optimization[ticker]).days
        
        if days_since_optimization >= self.optimization_interval:
            logger.info(f"{ticker} 전략 최적화 필요 (마지막 최적화 후 {days_since_optimization}일 경과)")
            return True
            
        return False
    
    def retrain_models(self, tickers=None):
        """ML 모델 재학습"""
        if tickers is None:
            tickers = self.tickers
            
        for ticker in tickers:
            if self.check_retraining_needed(ticker):
                logger.info(f"{ticker} ML 모델 재학습 시작")
                
                manager = self.ml_managers.get(ticker)
                if manager:
                    # 모델 업데이트
                    manager.update_models()
                    
                    # 마지막 학습 시간 업데이트
                    self.last_retraining[ticker] = datetime.now()
                    
                    logger.info(f"{ticker} ML 모델 재학습 완료")
        
        return True     

    def optimize_strategies(self, tickers=None):
        """전략 최적화"""
        if tickers is None:
            tickers = self.tickers
            
        for ticker in tickers:
            if self.check_optimization_needed(ticker):
                logger.info(f"{ticker} 전략 최적화 시작")
                
                optimizer = self.optimizers.get(ticker)
                if optimizer:
                    # 데이터 준비
                    optimizer.prepare_data()
                    
                    # 최적화 실행
                    best_params = optimizer.run_optimization(n_trials=30)
                    
                    if best_params:
                        # 기존 전략 업데이트
                        self.strategies[ticker]['default'] = TechnicalStrategy(f"{ticker} 최적화 전략", best_params)
                        
                        # 최적화 시간 업데이트
                        self.last_optimization[ticker] = datetime.now()
                        
                        logger.info(f"{ticker} 전략 최적화 완료")
        
        return True           
    
    def _management_loop(self):
        """관리 루프"""
        while self.running:
            try:
                # 시장 상태에 따른 전략 업데이트
                self.update_strategies()
                
                # ML 모델 재학습 확인
                self.retrain_models()
                
                # 전략 최적화 확인
                self.optimize_strategies()
                
                # 대기
                time.sleep(self.trading_interval)
                
            except Exception as e:
                logger.error(f"관리 오류: {e}")
                time.sleep(60)  # 오류 발생 시 잠시 대기 후 재시도
    
    def stop_auto_management(self):
        """자동 관리 중지"""
        self.running = False
        
        # 상태 감지 중지
        for ticker, detector in self.state_detectors.items():
            detector.stop_auto_detection()
            
        # ML 모델 관리 중지
        for ticker, manager in self.ml_managers.items():
            manager.stop()
            
        logger.info("자동 관리 중지됨")
    
    def get_system_status(self):
        """시스템 상태 정보"""
        status = {
            'running': self.running,
            'tickers': self.tickers,
            'current_states': {},
            'active_strategies': {},
            'portfolio': None
        }
        
        # 현재 상태 정보
        for ticker, detector in self.state_detectors.items():
            if detector.current_state:
                status['current_states'][ticker] = detector.current_state.get_state_summary()
        
        # 활성 전략 정보
        for ticker, strategy in self.active_strategies.items():
            status['active_strategies'][ticker] = {
                'name': strategy.get_name(),
                'type': type(strategy).__name__
            }
        
        # 포트폴리오 정보
        if self.executor and hasattr(self.executor, 'paper_account'):
            account = self.executor.paper_account
            
            # 현재가 정보
            current_prices = {}
            for ticker in self.tickers:
                # 수정된 부분: order_manager 사용
                if hasattr(self.executor, 'order_manager') and self.executor.order_manager:
                    price = self.executor.order_manager.get_current_price(ticker)
                    if price:
                        current_prices[ticker] = price
            
            # 포트폴리오 요약
            status['portfolio'] = account.get_portfolio_summary(current_prices)
        
        return status
    
    # ========== 여기에 추가 ==========
    def evaluate_and_manage_funds(self):
        """일일 포트폴리오 평가 및 자금 관리"""
        # 현재 포트폴리오 가치 계산
        portfolio = self.get_system_status()['portfolio']
        
        if not portfolio:
            logger.warning("포트폴리오 정보를 가져올 수 없습니다")
            return
            
        # 텔레그램 포트폴리오 정보 업데이트
        if hasattr(self, 'telegram_notifier') and self.telegram_notifier:
            self.telegram_notifier.update_portfolio_summary(portfolio)
        
        # 자금 관리 로직이 있는 경우
        if hasattr(self, 'fund_manager'):
            # 펀드 매니저에 성과 업데이트
            profit_ratio = self.fund_manager.update_portfolio_performance(portfolio['total_value'])
            
            # 수익 실현 확인
            extraction = self.fund_manager.check_profit_extraction()
            if extraction:
                # 수익 실현 작업 수행
                withdrawal_amount = extraction['amount']
                logger.info(f"수익 실현: {withdrawal_amount:,.0f}원 (수익률: {profit_ratio:.2f}%)")
                
                # 텔레그램 알림
                if hasattr(self, 'telegram_notifier') and self.telegram_notifier:
                    self.telegram_notifier.send_message(
                        f"💰 *수익 실현 완료*\n"
                        f"금액: {withdrawal_amount:,.0f}원\n"
                        f"수익률: {profit_ratio:.2f}%\n"
                        f"남은 자본: {extraction['new_capital']:,.0f}원",
                        parse_mode='Markdown'
                    )
            
            # 추가 투자 확인
            reinvestment = self.fund_manager.check_reinvestment()
            if reinvestment:
                # 추가 투자 작업 수행
                deposit_amount = reinvestment['amount']
                logger.info(f"추가 투자: {deposit_amount:,.0f}원 (현재 수익률: {profit_ratio:.2f}%)")
                
                # 텔레그램 알림
                if hasattr(self, 'telegram_notifier') and self.telegram_notifier:
                    self.telegram_notifier.send_message(
                        f"💼 *추가 투자 실행*\n"
                        f"금액: {deposit_amount:,.0f}원\n"
                        f"현재 수익률: {profit_ratio:.2f}%\n"
                        f"새 자본금: {reinvestment['new_capital']:,.0f}원",
                        parse_mode='Markdown'
                    )
        
        # 리밸런싱 확인
        if hasattr(self, 'portfolio_rebalancer'):
            current_allocations = {
                ticker: holding['value']
                for ticker, holding in portfolio['holdings'].items()
            }
            
            if self.portfolio_rebalancer.check_rebalance_needed(current_allocations):
                # 리밸런싱 계산
                rebalance_orders = self.portfolio_rebalancer.calculate_rebalance_orders(
                    current_allocations, 
                    {ticker: portfolio['holdings'][ticker]['current_price'] for ticker in current_allocations}
                )
                
                # 리밸런싱 실행
                if rebalance_orders:
                    for order in rebalance_orders:
                        ticker = order['ticker']
                        order_type = order['type']
                        amount = order['amount']
                        
                        if order_type == 'buy':
                            # 매수 주문
                            self.executor.order_manager.buy_market_order(ticker, amount)
                        else:
                            # 매도 주문
                            amount_coin = amount / order['price']
                            self.executor.order_manager.sell_market_order(ticker, amount_coin)
                    
                    logger.info("포트폴리오 리밸런싱 실행")
                    
                    # 텔레그램 알림
                    if hasattr(self, 'telegram_notifier') and self.telegram_notifier:
                        rebalance_msg = "\n".join([
                            f"- {order['ticker']}: {order['type']} {order['amount']:,.0f}원"
                            for order in rebalance_orders
                        ])
                        
                        self.telegram_notifier.send_message(
                            f"⚖️ *포트폴리오 리밸런싱 실행*\n"
                            f"조정 내역:\n{rebalance_msg}",
                            parse_mode='Markdown'
                        )
    

    def _schedule_portfolio_evaluation(self):
        """일일 포트폴리오 평가 예약"""
        import schedule
        
        # 매일 오전 9시와 오후 6시에 평가
        schedule.every().day.at("09:00").do(self.evaluate_and_manage_funds)
        schedule.every().day.at("18:00").do(self.evaluate_and_manage_funds)
        
        logger.info("일일 포트폴리오 평가 예약됨 (09:00, 18:00)")
    

    def optimize_strategies(self, tickers=None):
        """전략 최적화"""
        if tickers is None:
            tickers = self.tickers
            
        for ticker in tickers:
            if self.check_optimization_needed(ticker):
                logger.info(f"{ticker} 전략 최적화 시작")
                
                optimizer = self.optimizers.get(ticker)
                if optimizer:
                    # 데이터 준비
                    optimizer.prepare_data()
                    
                    # 최적화 실행
                    best_params = optimizer.run_optimization(n_trials=30)
                    
                    if best_params:
                        # 기존 전략 업데이트
                        self.strategies[ticker]['default'] = TechnicalStrategy(f"{ticker} 최적화 전략", best_params)
                        
                        # 최적화 시간 업데이트
                        self.last_optimization[ticker] = datetime.now()
                        
                        logger.info(f"{ticker} 전략 최적화 완료")
        
        return True
    
    def start_trading(self):
        """거래 시작"""
        if not self.running:
            self.running = True
            
            # 상태 감지기 시작
            for ticker, detector in self.state_detectors.items():
                detector.start_auto_detection()
                
            # ML 모델 관리자 시작
            for ticker, manager in self.ml_managers.items():
                manager.start()
                
            # 실행기 시작
            if self.executor:
                self.executor.start_trading(self.tickers)
                
            # 관리 루프 시작 (백그라운드 스레드)
            management_thread = Thread(target=self._management_loop)
            management_thread.daemon = True
            management_thread.start()
            self.management_threads['main'] = management_thread
            
            logger.info("자동 거래 시작됨")
            return True
        
        logger.warning("이미 거래가 실행 중입니다")
        return False
    
    def stop_trading(self):
        """거래 중지"""
        if self.running:
            self.running = False
            
            # 모든 관리 스레드 중지
            self.stop_auto_management()
            
            # 실행기 중지
            if self.executor:
                self.executor.stop_trading()
                
            logger.info("자동 거래 중지됨")
            return True
        
        logger.warning("거래가 이미 중지되었습니다")
        return False

    def setup_sentiment_analysis(self):
        """감정 분석 모듈 설정"""
        if not self.sentiment_enabled:
            return
            
        try:
            # 감정 분석을 위한 각 코인별 수집기 설정
            for ticker in self.tickers:
                self.sentiment_collectors[ticker] = SentimentDataCollector()
                
                # 감정 기반 전략 생성
                sentiment_strategy = SentimentStrategy(f"{ticker} 감정 전략")
                
                # 전략 목록에 추가
                if ticker not in self.strategies:
                    self.strategies[ticker] = {}
                self.strategies[ticker]['sentiment'] = sentiment_strategy
                
                logger.info(f"{ticker} 감정 분석 설정 완료")
                
        except Exception as e:
            logger.error(f"감정 분석 설정 오류: {e}")
            self.sentiment_enabled = False

    def setup_fund_management(self):
        """자금 관리 모듈 설정"""
        if not self.fund_manager_enabled:
            return
            
        try:
            # 포트폴리오 목표 할당 설정 (균등 분배)
            target_allocation = {}
            coin_weight = 0.8 / len(self.tickers)  # 80%를 코인에 할당, 20%는 현금
            
            for ticker in self.tickers:
                target_allocation[ticker] = coin_weight
                
            self.portfolio_rebalancer.set_target_allocation(target_allocation)
            
            logger.info("자금 관리 모듈 설정 완료")
            
        except Exception as e:
            logger.error(f"자금 관리 설정 오류: {e}")
            self.fund_manager_enabled = False

    def _schedule_portfolio_evaluation(self):
        """일일 포트폴리오 평가 예약"""
        import schedule
        
        # 매일 오전 9시와 오후 6시에 평가
        schedule.every().day.at("09:00").do(self.evaluate_and_manage_funds)
        schedule.every().day.at("18:00").do(self.evaluate_and_manage_funds)
        
        logger.info("일일 포트폴리오 평가 예약됨 (09:00, 18:00)")

# 사용 예시
if __name__ == "__main__":
    system = AdvancedTradingSystem(initial_balance=1000000)
    
    # 코인 선택
    system.select_coins(method='balanced', count=3)
    
    # 시스템 초기화
    system.initialize_system()
    
    # 거래 시작
    system.start_trading()
    
    try:
        # 메인 스레드는 정기적으로 상태 출력
        while True:
            status = system.get_system_status()
            print(f"\n=== 시스템 상태 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
            print(f"실행 중: {'예' if status['running'] else '아니오'}")
            print(f"거래 코인: {', '.join(status['tickers'])}")
            
            print("\n현재 시장 상태:")
            for ticker, state in status['current_states'].items():
                print(f"  {ticker}: {state['characteristics']}")
                
            print("\n활성 전략:")
            for ticker, strategy in status['active_strategies'].items():
                print(f"  {ticker}: {strategy['name']}")
                
            if status['portfolio']:
                portfolio = status['portfolio']
                profit_percent = portfolio['total_profit_percent']
                print(f"\n포트폴리오: {portfolio['total_value']:,.0f}원 (수익률: {profit_percent:.2f}%)")
                print(f"보유 현금: {portfolio['balance']:,.0f}원")
                
                for ticker, holding in portfolio['holdings'].items():
                    print(f"  {ticker}: {holding['amount']:.8f} 개, " +
                          f"평균매수가 {holding['avg_price']:,.0f}원, " +
                          f"현재가 {holding['current_price']:,.0f}원, " +
                          f"수익률 {holding['profit_percent']:.2f}%")
            
            print("="*60)
            
            time.sleep(300)  # 5분마다 업데이트
            
    except KeyboardInterrupt:
        system.stop_trading()
        print("프로그램이 종료되었습니다.")
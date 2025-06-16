
#!/usr/bin/env python3
"""
개선된 자동매매 시스템 메인 스크립트
"""

import os
import time
import signal
import sys
from datetime import datetime
from data.improved_coin_selector import ImprovedCoinSelector
from strategy.improved_ml_strategy import ImprovedMLStrategy
from trading.risk_manager import RiskManager
from utils.system_monitor import SystemMonitor
from utils.logger import setup_logger

logger = setup_logger("improved_main")

class ImprovedTradingSystem:
    """개선된 자동매매 시스템"""
    
    def __init__(self, initial_balance=20000000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.running = False
        
        # 핵심 모듈들
        self.coin_selector = ImprovedCoinSelector()
        self.risk_manager = RiskManager()
        self.monitor = SystemMonitor()
        
        # 선택된 코인과 전략
        self.selected_coins = []
        self.strategies = {}
        self.positions = {}
        
        # 성능 추적
        self.trade_history = []
        self.last_rebalance = datetime.now()
        
    def initialize_system(self):
        """시스템 초기화"""
        try:
            logger.info("🚀 개선된 자동매매 시스템 초기화 시작")
            
            # 1. 코인 선택
            logger.info("1️⃣ 고품질 코인 선택 중...")
            self.selected_coins, coin_scores = self.coin_selector.select_quality_coins(target_count=3)
            
            if not self.selected_coins:
                raise Exception("선택된 코인이 없습니다")
            
            logger.info(f"선택된 코인: {', '.join(self.selected_coins)}")
            
            # 2. 각 코인별 ML 전략 초기화
            logger.info("2️⃣ ML 전략 초기화 중...")
            for ticker in self.selected_coins:
                try:
                    strategy = ImprovedMLStrategy(ticker)
                    self.strategies[ticker] = strategy
                    self.positions[ticker] = {'quantity': 0, 'avg_price': 0}
                    logger.info(f"{ticker} 전략 초기화 완료")
                except Exception as e:
                    logger.error(f"{ticker} 전략 초기화 실패: {e}")
            
            # 3. 모델 훈련
            logger.info("3️⃣ ML 모델 훈련 시작...")
            self._train_models()
            
            logger.info("✅ 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            return False
    
    def _train_models(self):
        """모든 모델 훈련"""
        from data.collector import UpbitDataCollector
        
        data_collector = UpbitDataCollector()
        
        for ticker in self.selected_coins:
            try:
                logger.info(f"{ticker} 모델 훈련 시작...")
                
                # 데이터 수집
                df = data_collector.get_ohlcv(ticker, interval='day', count=200)
                if df is None or len(df) < 100:
                    logger.warning(f"{ticker} 데이터 부족으로 모델 훈련 스킵")
                    continue
                
                # 특성 준비 및 모델 훈련
                strategy = self.strategies[ticker]
                features, target = strategy.prepare_features(df)
                
                if features is not None and target is not None:
                    success = strategy.train_ensemble_model(features, target)
                    if success:
                        logger.info(f"{ticker} 모델 훈련 완료")
                    else:
                        logger.warning(f"{ticker} 모델 훈련 실패")
                else:
                    logger.warning(f"{ticker} 특성 준비 실패")
                    
            except Exception as e:
                logger.error(f"{ticker} 모델 훈련 오류: {e}")
    
    def start_trading(self):
        """거래 시작"""
        if self.running:
            logger.warning("시스템이 이미 실행 중입니다")
            return
        
        self.running = True
        logger.info("🎯 자동매매 시작")
        
        # 신호 처리 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running:
                self._trading_cycle()
                time.sleep(3600)  # 1시간 대기
                
        except KeyboardInterrupt:
            logger.info("사용자 중단 요청")
        except Exception as e:
            logger.error(f"거래 중 오류: {e}")
        finally:
            self.stop_trading()
    
    def _trading_cycle(self):
        """거래 사이클 실행"""
        try:
            logger.info(f"🔄 거래 사이클 시작 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            
            from data.collector import UpbitDataCollector
            data_collector = UpbitDataCollector()
            
            # 각 코인별 신호 생성 및 거래 실행
            for ticker in self.selected_coins:
                try:
                    # 최신 데이터 수집
                    df = data_collector.get_ohlcv(ticker, interval='day', count=100)
                    if df is None:
                        continue
                    
                    # 현재 가격
                    current_price = df['close'].iloc[-1]
                    
                    # 거래 신호 생성
                    strategy = self.strategies[ticker]
                    signal = strategy.get_signal(df)
                    
                    # 리스크 관리 적용
                    position_info = self.positions[ticker]
                    should_trade = self.risk_manager.should_execute_trade(
                        ticker, signal, current_price, position_info
                    )
                    
                    if should_trade:
                        self._execute_trade(ticker, signal, current_price, df)
                    
                    logger.info(f"{ticker}: 신호={signal}, 가격={current_price:,.0f}, 거래={should_trade}")
                    
                except Exception as e:
                    logger.error(f"{ticker} 거래 처리 오류: {e}")
            
            # 성능 모니터링
            self._update_performance()
            
            # 정기 리밸런싱 (7일마다)
            if (datetime.now() - self.last_rebalance).days >= 7:
                self._rebalance_portfolio()
                self.last_rebalance = datetime.now()
            
        except Exception as e:
            logger.error(f"거래 사이클 오류: {e}")
    
    def _execute_trade(self, ticker, signal, current_price, df):
        """거래 실행 (페이퍼 트레이딩)"""
        try:
            position = self.positions[ticker]
            
            if signal in ['buy', 'strong_buy'] and position['quantity'] == 0:
                # 매수
                volatility = df['close'].pct_change().std()
                quantity = self.risk_manager.calculate_position_size(
                    signal, current_price, self.current_balance * 0.8, volatility
                )
                
                if quantity > 0:
                    cost = quantity * current_price
                    if cost <= self.current_balance:
                        # 거래 실행
                        self.current_balance -= cost
                        position['quantity'] = quantity
                        position['avg_price'] = current_price
                        
                        # 거래 기록
                        trade_record = {
                            'timestamp': datetime.now().isoformat(),
                            'ticker': ticker,
                            'type': 'buy',
                            'quantity': quantity,
                            'price': current_price,
                            'value': cost,
                            'signal': signal
                        }
                        self.trade_history.append(trade_record)
                        
                        logger.info(f"💰 {ticker} 매수: {quantity:.4f}개 @ {current_price:,.0f}원 (총 {cost:,.0f}원)")
            
            elif signal in ['sell', 'strong_sell'] and position['quantity'] > 0:
                # 매도
                quantity = position['quantity']
                revenue = quantity * current_price
                
                # 거래 실행
                self.current_balance += revenue
                profit = revenue - (quantity * position['avg_price'])
                
                # 포지션 정리
                position['quantity'] = 0
                position['avg_price'] = 0
                
                # 거래 기록
                trade_record = {
                    'timestamp': datetime.now().isoformat(),
                    'ticker': ticker,
                    'type': 'sell',
                    'quantity': quantity,
                    'price': current_price,
                    'value': revenue,
                    'profit': profit,
                    'signal': signal
                }
                self.trade_history.append(trade_record)
                
                logger.info(f"💸 {ticker} 매도: {quantity:.4f}개 @ {current_price:,.0f}원 "
                           f"(수익: {profit:+,.0f}원, {profit/(quantity * position['avg_price']):+.2%})")
                
        except Exception as e:
            logger.error(f"{ticker} 거래 실행 오류: {e}")
    
    def _update_performance(self):
        """성능 업데이트"""
        try:
            # 현재 포트폴리오 가치 계산
            portfolio_value = self.current_balance
            
            from data.collector import UpbitDataCollector
            data_collector = UpbitDataCollector()
            
            for ticker, position in self.positions.items():
                if position['quantity'] > 0:
                    try:
                        df = data_collector.get_ohlcv(ticker, interval='day', count=1)
                        if df is not None:
                            current_price = df['close'].iloc[-1]
                            portfolio_value += position['quantity'] * current_price
                    except:
                        pass
            
            # ML 정확도 계산
            avg_accuracy = 0
            accuracy_count = 0
            for strategy in self.strategies.values():
                if hasattr(strategy, 'last_performance'):
                    perf = strategy.last_performance.get('cv_scores', {})
                    if perf:
                        avg_accuracy += sum(perf.values()) / len(perf)
                        accuracy_count += 1
            
            ml_accuracy = avg_accuracy / accuracy_count if accuracy_count > 0 else None
            
            # 성능 로깅
            self.monitor.log_performance(portfolio_value, self.trade_history, ml_accuracy)
            
            # 성능 요약 출력
            summary = self.monitor.get_performance_summary()
            if summary:
                total_return = (portfolio_value - self.initial_balance) / self.initial_balance
                logger.info(f"📊 포트폴리오: {portfolio_value:,.0f}원 ({total_return:+.2%}), "
                           f"샤프비율: {summary.get('sharpe_ratio', 0):.2f}, "
                           f"최대손실: {summary.get('max_drawdown', 0):.2%}")
            
        except Exception as e:
            logger.error(f"성능 업데이트 오류: {e}")
    
    def _rebalance_portfolio(self):
        """포트폴리오 리밸런싱"""
        try:
            logger.info("🔄 포트폴리오 리밸런싱 시작")
            
            # 새로운 코인 선택
            new_coins, _ = self.coin_selector.select_quality_coins(target_count=3)
            
            # 기존 코인과 비교
            coins_to_remove = set(self.selected_coins) - set(new_coins)
            coins_to_add = set(new_coins) - set(self.selected_coins)
            
            if coins_to_remove or coins_to_add:
                logger.info(f"코인 변경: 제거={list(coins_to_remove)}, 추가={list(coins_to_add)}")
                
                # 제거할 코인 매도
                for ticker in coins_to_remove:
                    if self.positions[ticker]['quantity'] > 0:
                        # 강제 매도 (시장가)
                        logger.info(f"리밸런싱으로 {ticker} 매도")
                        # 실제 매도 로직은 _execute_trade와 유사하게 구현
                
                # 코인 목록 업데이트
                self.selected_coins = new_coins
                
                # 새 코인 전략 초기화
                for ticker in coins_to_add:
                    self.strategies[ticker] = ImprovedMLStrategy(ticker)
                    self.positions[ticker] = {'quantity': 0, 'avg_price': 0}
                
                # 모델 재훈련
                self._train_models()
            
        except Exception as e:
            logger.error(f"리밸런싱 오류: {e}")
    
    def _signal_handler(self, signum, frame):
        """시스템 종료 신호 처리"""
        logger.info(f"종료 신호 수신: {signum}")
        self.running = False
    
    def stop_trading(self):
        """거래 중지"""
        self.running = False
        
        # 최종 성과 요약
        final_value = self.current_balance
        for ticker, position in self.positions.items():
            if position['quantity'] > 0:
                final_value += position['quantity'] * position['avg_price']
        
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        logger.info("📈 최종 거래 결과")
        logger.info(f"초기 자본: {self.initial_balance:,}원")
        logger.info(f"최종 자본: {final_value:,.0f}원")
        logger.info(f"총 수익률: {total_return:+.2%}")
        logger.info(f"총 거래횟수: {len(self.trade_history)}회")
        
        logger.info("🛑 자동매매 시스템 종료")

def main():
    """메인 함수"""
    try:
        # 시스템 초기화
        system = ImprovedTradingSystem(initial_balance=20_000_000)
        
        if not system.initialize_system():
            logger.error("시스템 초기화 실패")
            return
        
        # 거래 시작
        system.start_trading()
        
    except Exception as e:
        logger.error(f"메인 프로세스 오류: {e}")
    finally:
        logger.info("프로그램 종료")

if __name__ == "__main__":
    main()

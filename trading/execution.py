from config.settings import TRADING_MODE
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

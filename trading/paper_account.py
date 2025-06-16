import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger("paper_account")

class PaperAccount:
    """가상 거래 계정 관리"""
    
    def __init__(self, initial_balance=1000000):
        """초기화"""
        self.initial_balance = initial_balance
        self.balance = initial_balance  # 원화 잔고
        self.holdings = {}  # 코인 보유량 (ticker: amount)
        self.buy_prices = {}  # 코인별 매수 가격 (ticker: price)
        self.orders = []  # 주문 내역
        self.trade_history = []  # 거래 내역
        self.limit_orders = {}  # 지정가 주문 (order_id: order_info)
        
    def get_balance(self):
        """현재 원화 잔고 조회"""
        return self.balance
        
    def get_holdings(self):
        """코인 보유 현황 조회"""
        return self.holdings
        
    def get_portfolio_value(self, current_prices):
        """포트폴리오 총 가치 계산"""
        total_value = self.balance
        
        for ticker, amount in self.holdings.items():
            if ticker in current_prices and amount > 0:
                total_value += amount * current_prices[ticker]
                
        return total_value
    
    def get_avg_buy_price(self, ticker):
        """코인별 평균 매수가 조회"""
        if ticker in self.buy_prices:
            return self.buy_prices[ticker]
        return 0
        
    def buy(self, ticker, price, amount, commission=0.0005):
        """매수 실행"""
        timestamp = datetime.now()
        cost = price * amount
        fee = cost * commission
        total_cost = cost + fee
        
        # 잔고 확인
        if total_cost > self.balance:
            logger.warning(f"잔고 부족: 필요 {total_cost}, 보유 {self.balance}")
            return False
            
        # 거래 기록
        self.balance -= total_cost
        
        if ticker in self.holdings:
            # 기존 매수가와 현재 매수가의 가중 평균 계산
            current_amount = self.holdings[ticker]
            current_value = current_amount * self.buy_prices.get(ticker, 0)
            new_value = amount * price
            
            self.holdings[ticker] += amount
            total_amount = self.holdings[ticker]
            
            # 평균 매수가 업데이트
            if total_amount > 0:
                self.buy_prices[ticker] = (current_value + new_value) / total_amount
        else:
            self.holdings[ticker] = amount
            self.buy_prices[ticker] = price
            
        trade = {
            'timestamp': timestamp,
            'ticker': ticker,
            'type': 'buy',
            'price': price,
            'amount': amount,
            'fee': fee,
            'total': total_cost
        }
        
        self.trade_history.append(trade)
        logger.info(f"매수 완료: {ticker}, 가격: {price}, 수량: {amount}, 총액: {total_cost}")
        
        return True
    
    def sell(self, ticker, price, amount, commission=0.0005):
        """매도 실행"""
        timestamp = datetime.now()
        
        # 보유량 확인 (근사 비교로 소수점 문제 해결)
        current_amount = self.holdings.get(ticker, 0)
        amount_diff = abs(current_amount - amount)
        amount_relative_diff = amount_diff / max(current_amount, 1e-10)
        
        # 보유량보다 큰 경우 처리
        if amount > current_amount and amount_relative_diff > 0.001:  # 0.1% 이상 차이
            logger.warning(f"보유량 부족: 필요 {amount}, 보유 {current_amount}, 보유량으로 조정")
            amount = current_amount
        
        # 매우 근소한 차이는 전액 매도로 처리 (소수점 문제 해결)
        if amount_relative_diff < 0.001:  # 0.1% 미만 차이
            amount = current_amount
            logger.debug(f"소수점 차이 조정: {amount_diff} (전체 매도로 처리)")
        
        # 거래 기록
        value = price * amount
        fee = value * commission
        net_value = value - fee
        
        self.balance += net_value
        self.holdings[ticker] -= amount
        
        # 보유량이 아주 작은 경우 0으로 처리 (소수점 오차 방지)
        if abs(self.holdings[ticker]) < 1e-8:
            self.holdings[ticker] = 0
        
        # 모두 매도한 경우 매수가 정보 삭제
        if self.holdings[ticker] <= 0:
            self.holdings.pop(ticker, None)
            self.buy_prices.pop(ticker, None)
        
        trade = {
            'timestamp': timestamp,
            'ticker': ticker,
            'type': 'sell',
            'price': price,
            'amount': amount,
            'fee': fee,
            'total': net_value
        }
        
        self.trade_history.append(trade)
        logger.info(f"매도 완료: {ticker}, 가격: {price}, 수량: {amount}, 총액: {net_value}")
        
        return True
        
    def get_trade_history(self):
        """거래 내역 조회"""
        return pd.DataFrame(self.trade_history)
        
    def register_limit_order(self, order_type, ticker, price, quantity, order_id):
        """지정가 주문 등록"""
        self.limit_orders[order_id] = {
            'type': order_type,
            'ticker': ticker,
            'price': price,
            'quantity': quantity,
            'timestamp': datetime.now()
        }
        return order_id
    
    def cancel_limit_order(self, order_id):
        """지정가 주문 취소"""
        if order_id in self.limit_orders:
            del self.limit_orders[order_id]
            return True
        return False
        
    def get_unrealized_profit(self, ticker, current_price):
        """미실현 손익 계산"""
        if ticker not in self.holdings or ticker not in self.buy_prices:
            return 0
            
        amount = self.holdings[ticker]
        avg_price = self.buy_prices[ticker]
        
        return amount * (current_price - avg_price)
        
    def get_profit_percent(self, ticker, current_price):
        """수익률 계산"""
        if ticker not in self.buy_prices or self.buy_prices[ticker] == 0:
            return 0
            
        avg_price = self.buy_prices[ticker]
        return (current_price / avg_price - 1) * 100
        
    def reset(self):
        """계정 초기화"""
        self.balance = self.initial_balance
        self.holdings = {}
        self.buy_prices = {}
        self.orders = []
        self.trade_history = []
        self.limit_orders = {}
        logger.info(f"계정 초기화: 잔고 {self.initial_balance}")
        
    def get_position_value(self, ticker, current_price):
        """특정 코인의 포지션 가치"""
        if ticker not in self.holdings:
            return 0
            
        return self.holdings[ticker] * current_price
    
    def get_total_invested(self):
        """총 투자 금액"""
        invested = 0
        for ticker, amount in self.holdings.items():
            if ticker in self.buy_prices:
                invested += amount * self.buy_prices[ticker]
        return invested
        
    def get_portfolio_summary(self, current_prices):
        """포트폴리오 요약"""
        summary = {
            'balance': self.balance,
            'holdings': {},
            'total_value': self.balance,
            'total_invested': 0,
            'total_profit': 0
        }
        
        for ticker, amount in self.holdings.items():
            if ticker in current_prices and amount > 0:
                current_price = current_prices[ticker]
                avg_price = self.buy_prices.get(ticker, 0)
                position_value = amount * current_price
                invested = amount * avg_price
                profit = position_value - invested
                profit_percent = (current_price / avg_price - 1) * 100 if avg_price > 0 else 0
                
                summary['holdings'][ticker] = {
                    'amount': amount,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'value': position_value,
                    'invested': invested,
                    'profit': profit,
                    'profit_percent': profit_percent
                }
                
                summary['total_value'] += position_value
                summary['total_invested'] += invested
                summary['total_profit'] += profit
        
        # 전체 수익률
        if summary['total_invested'] > 0:
            summary['total_profit_percent'] = (summary['total_profit'] / summary['total_invested']) * 100
        else:
            summary['total_profit_percent'] = 0
            
        return summary
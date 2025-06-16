
import logging
import numpy as np
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger("risk_manager")

class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, max_position_size=0.3, stop_loss=0.05, take_profit=0.15):
        """
        Args:
            max_position_size (float): 최대 포지션 크기 (전체 자본 대비)
            stop_loss (float): 손절매 비율
            take_profit (float): 익절매 비율
        """
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_tracker = {}
        
    def calculate_position_size(self, signal_strength, current_price, available_balance, volatility):
        """포지션 크기 계산"""
        try:
            # 기본 포지션 크기
            base_size = available_balance * self.max_position_size
            
            # 신호 강도에 따른 조정
            signal_multiplier = {
                'strong_buy': 1.0,
                'buy': 0.7,
                'strong_sell': 1.0,
                'sell': 0.7,
                'hold': 0.0
            }.get(signal_strength, 0.0)
            
            # 변동성에 따른 조정 (높은 변동성 = 작은 포지션)
            volatility_multiplier = max(0.3, 1.0 - (volatility * 2))
            
            # 최종 포지션 크기
            position_value = base_size * signal_multiplier * volatility_multiplier
            position_quantity = position_value / current_price
            
            logger.debug(f"포지션 크기 계산: {position_value:,.0f}원 ({position_quantity:.4f}개)")
            return position_quantity
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 오류: {e}")
            return 0
    
    def should_execute_trade(self, ticker, signal, current_price, position_info):
        """거래 실행 여부 결정"""
        try:
            # 포지션이 없는 경우
            if not position_info or position_info.get('quantity', 0) == 0:
                return signal in ['buy', 'strong_buy']
            
            # 포지션이 있는 경우
            entry_price = position_info.get('avg_price', current_price)
            quantity = position_info.get('quantity', 0)
            
            if quantity > 0:  # 롱 포지션
                profit_ratio = (current_price - entry_price) / entry_price
                
                # 손절매 체크
                if profit_ratio <= -self.stop_loss:
                    logger.info(f"{ticker} 손절매 실행: {profit_ratio:.2%}")
                    return True
                
                # 익절매 체크
                if profit_ratio >= self.take_profit:
                    logger.info(f"{ticker} 익절매 실행: {profit_ratio:.2%}")
                    return True
                
                # 매도 신호 체크
                if signal in ['sell', 'strong_sell']:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"{ticker} 거래 실행 판단 오류: {e}")
            return False
    
    def update_position(self, ticker, trade_type, quantity, price):
        """포지션 업데이트"""
        try:
            if ticker not in self.position_tracker:
                self.position_tracker[ticker] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'total_cost': 0,
                    'entry_time': None
                }
            
            position = self.position_tracker[ticker]
            
            if trade_type == 'buy':
                # 매수
                new_cost = position['total_cost'] + (quantity * price)
                new_quantity = position['quantity'] + quantity
                
                if new_quantity > 0:
                    position['avg_price'] = new_cost / new_quantity
                position['quantity'] = new_quantity
                position['total_cost'] = new_cost
                position['entry_time'] = datetime.now()
                
            elif trade_type == 'sell':
                # 매도
                sell_quantity = min(quantity, position['quantity'])
                sell_cost = sell_quantity * position['avg_price']
                
                position['quantity'] -= sell_quantity
                position['total_cost'] -= sell_cost
                
                if position['quantity'] <= 0:
                    position['quantity'] = 0
                    position['total_cost'] = 0
                    position['avg_price'] = 0
                    position['entry_time'] = None
            
            logger.debug(f"{ticker} 포지션 업데이트: {position}")
            
        except Exception as e:
            logger.error(f"{ticker} 포지션 업데이트 오류: {e}")
    
    def get_portfolio_risk(self, portfolio_value, positions):
        """포트폴리오 리스크 계산"""
        try:
            total_risk = 0
            
            for ticker, position in positions.items():
                if position.get('quantity', 0) > 0:
                    position_value = position['quantity'] * position['avg_price']
                    position_weight = position_value / portfolio_value
                    
                    # 집중도 리스크
                    concentration_risk = max(0, position_weight - self.max_position_size)
                    total_risk += concentration_risk
            
            return {
                'total_risk': total_risk,
                'risk_level': 'High' if total_risk > 0.2 else 'Medium' if total_risk > 0.1 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 리스크 계산 오류: {e}")
            return {'total_risk': 0, 'risk_level': 'Unknown'}

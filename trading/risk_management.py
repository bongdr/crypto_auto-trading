from datetime import datetime, timedelta
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

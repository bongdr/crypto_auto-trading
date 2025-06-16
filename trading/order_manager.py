from config.settings import TRADING_MODE, UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
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

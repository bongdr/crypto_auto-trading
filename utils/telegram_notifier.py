
import requests
import json
import logging
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger("telegram_notifier")

class TelegramNotifier:
    """텔레그램 알림 발송기"""
    
    def __init__(self, bot_token, chat_id):
        """초기화
        
        Args:
            bot_token (str): 텔레그램 봇 토큰
            chat_id (str): 채팅 ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        
        # 연결 테스트
        if self._test_connection():
            logger.info("텔레그램 알림 초기화 성공")
        else:
            logger.warning("텔레그램 연결 테스트 실패")
    
    def _test_connection(self):
        """연결 테스트"""
        try:
            response = requests.get(f"{self.api_url}/getMe", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"텔레그램 연결 테스트 오류: {e}")
            return False
    
    def send_message(self, message, parse_mode='Markdown'):
        """메시지 전송
        
        Args:
            message (str): 전송할 메시지
            parse_mode (str): 파싱 모드 (Markdown 또는 HTML)
        
        Returns:
            bool: 전송 성공 여부
        """
        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.debug("텔레그램 메시지 전송 성공")
                return True
            else:
                logger.error(f"텔레그램 메시지 전송 실패: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"텔레그램 메시지 전송 오류: {e}")
            return False
    
    def send_trade_alert(self, ticker, action, price, quantity, profit=None):
        """거래 알림 전송
        
        Args:
            ticker (str): 코인 티커
            action (str): 거래 동작 (buy/sell)
            price (float): 거래 가격
            quantity (float): 수량
            profit (float): 수익 (매도시)
        """
        try:
            emoji = "💰" if action == "buy" else "💸"
            action_text = "매수" if action == "buy" else "매도"
            
            message = f"{emoji} **거래 알림**\n"
            message += f"코인: `{ticker}`\n"
            message += f"동작: {action_text}\n"
            message += f"가격: {price:,.0f}원\n"
            message += f"수량: {quantity:.4f}개\n"
            message += f"금액: {price * quantity:,.0f}원\n"
            
            if profit is not None:
                profit_emoji = "📈" if profit > 0 else "📉"
                message += f"수익: {profit_emoji} {profit:+,.0f}원 ({profit/(price*quantity)*100:+.2f}%)\n"
            
            message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"거래 알림 전송 오류: {e}")
            return False
    
    def send_performance_report(self, portfolio_value, initial_value, trades_count, summary=None):
        """성능 보고서 전송
        
        Args:
            portfolio_value (float): 현재 포트폴리오 가치
            initial_value (float): 초기 투자금
            trades_count (int): 총 거래 횟수
            summary (dict): 추가 성과 지표
        """
        try:
            total_return = (portfolio_value - initial_value) / initial_value * 100
            emoji = "📈" if total_return > 0 else "📉"
            
            message = f"📊 **일일 성과 보고서**\n\n"
            message += f"현재 자산: {portfolio_value:,.0f}원\n"
            message += f"초기 투자: {initial_value:,.0f}원\n"
            message += f"수익률: {emoji} {total_return:+.2f}%\n"
            message += f"총 거래: {trades_count}회\n"
            
            if summary:
                message += f"\n**추가 지표:**\n"
                if 'sharpe_ratio' in summary:
                    message += f"샤프 비율: {summary['sharpe_ratio']:.2f}\n"
                if 'max_drawdown' in summary:
                    message += f"최대 손실: {summary['max_drawdown']:.2%}\n"
                if 'volatility' in summary:
                    message += f"변동성: {summary['volatility']:.2%}\n"
            
            message += f"\n보고 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"성과 보고서 전송 오류: {e}")
            return False
    
    def send_system_alert(self, alert_type, message_text):
        """시스템 알림 전송
        
        Args:
            alert_type (str): 알림 유형 (warning, error, info)
            message_text (str): 알림 내용
        """
        try:
            emoji_map = {
                'warning': '⚠️',
                'error': '🚨',
                'info': 'ℹ️',
                'success': '✅'
            }
            
            emoji = emoji_map.get(alert_type, 'ℹ️')
            
            message = f"{emoji} **시스템 알림**\n\n"
            message += f"{message_text}\n\n"
            message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"시스템 알림 전송 오류: {e}")
            return False

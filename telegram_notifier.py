import requests
import time
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from threading import Thread
import schedule

logger = logging.getLogger("telegram_notifier")

class TelegramNotifier:
    """텔레그램을 통한 알림 기능"""
    
    def __init__(self, token, chat_id, log_level='INFO'):
        """초기화
        
        Args:
            token (str): 텔레그램 봇 토큰
            chat_id (str): 텔레그램 채팅 ID
            log_level (str): 로깅 레벨
        """
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.message_queue = []
        self.running = False
        self.daily_trades = []  # 일일 거래 기록
        self.daily_summaries = []  # 일일 요약 기록
        
        # 로깅 설정
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
        
        logger.setLevel(numeric_level)
        
        # 봇 연결 테스트
        self._test_connection()
    
    def _test_connection(self):
        """봇 연결 테스트"""
        try:
            response = requests.get(f"{self.base_url}/getMe")
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info['ok']:
                    logger.info(f"텔레그램 봇 연결 성공: @{bot_info['result']['username']}")
                    return True
                else:
                    logger.error(f"텔레그램 봇 연결 실패: {bot_info}")
            else:
                logger.error(f"텔레그램 API 응답 오류: {response.status_code}")
        except Exception as e:
            logger.error(f"텔레그램 연결 테스트 중 오류: {e}")
        return False
    
    def send_message(self, message, parse_mode='Markdown'):
        """텔레그램 메시지 전송
        
        Args:
            message (str): 전송할 메시지
            parse_mode (str): 파싱 모드 ('Markdown' 또는 'HTML')
            
        Returns:
            bool: 메시지 전송 성공 여부
        """
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                result = response.json()
                if result['ok']:
                    logger.debug("메시지 전송 성공")
                    return True
                else:
                    logger.error(f"메시지 전송 실패: {result}")
            else:
                logger.error(f"텔레그램 API 응답 오류: {response.status_code}")
                
            return False
            
        except Exception as e:
            logger.error(f"메시지 전송 중 오류: {e}")
            return False
    
    def send_chart(self, chart_path, caption=None):
        """차트 이미지 전송
        
        Args:
            chart_path (str): 차트 이미지 파일 경로
            caption (str): 이미지 설명
            
        Returns:
            bool: 이미지 전송 성공 여부
        """
        if not os.path.exists(chart_path):
            logger.error(f"차트 파일이 존재하지 않습니다: {chart_path}")
            return False
            
        try:
            url = f"{self.base_url}/sendPhoto"
            data = {
                'chat_id': self.chat_id
            }
            
            if caption:
                data['caption'] = caption
                data['parse_mode'] = 'Markdown'
                
            with open(chart_path, 'rb') as photo:
                response = requests.post(url, data=data, files={'photo': photo})
                
            if response.status_code == 200:
                result = response.json()
                if result['ok']:
                    logger.debug("차트 전송 성공")
                    return True
                else:
                    logger.error(f"차트 전송 실패: {result}")
            else:
                logger.error(f"텔레그램 API 응답 오류: {response.status_code}")
                
            return False
            
        except Exception as e:
            logger.error(f"차트 전송 중 오류: {e}")
            return False
    
    def notify_trade(self, trade_info):
        """거래 알림
        
        Args:
            trade_info (dict): 거래 정보
        """
        action = trade_info.get('action')
        ticker = trade_info.get('ticker')
        price = trade_info.get('price', 0)
        amount = trade_info.get('amount', 0)
        
        # 각 거래 유형별 이모티콘과 메시지 형식
        if action == 'buy':
            emoji = "🟢"
            message = (
                f"{emoji} *{ticker} 매수 완료*\n"
                f"가격: {price:,.0f}원\n"
                f"수량: {amount:.8f}\n"
                f"총액: {price * amount:,.0f}원"
            )
        elif action == 'sell':
            emoji = "🔴"
            profit_percent = trade_info.get('profit_percent', 0)
            message = (
                f"{emoji} *{ticker} 매도 완료*\n"
                f"가격: {price:,.0f}원\n"
                f"수량: {amount:.8f}\n"
                f"총액: {price * amount:,.0f}원\n"
                f"수익률: {profit_percent:.2f}%"
            )
        elif action == 'stop_loss':
            emoji = "⚠️"
            profit_percent = trade_info.get('profit_percent', 0)
            message = (
                f"{emoji} *{ticker} 손절매 실행*\n"
                f"가격: {price:,.0f}원\n"
                f"수량: {amount:.8f}\n"
                f"손실률: {profit_percent:.2f}%"
            )
        elif action == 'take_profit':
            emoji = "💰"
            profit_percent = trade_info.get('profit_percent', 0)
            message = (
                f"{emoji} *{ticker} 익절 실행*\n"
                f"가격: {price:,.0f}원\n"
                f"수량: {amount:.8f}\n"
                f"수익률: {profit_percent:.2f}%"
            )
        else:
            return  # 알 수 없는 액션은 알림 보내지 않음
        
        # 메시지 전송
        self.send_message(message)
        
        # 일일 거래 기록에 추가
        trade_record = {
            'timestamp': datetime.now(),
            'action': action,
            'ticker': ticker,
            'price': price,
            'amount': amount,
            'profit_percent': trade_info.get('profit_percent', 0)
        }
        self.daily_trades.append(trade_record)
    
    def notify_state_change(self, ticker, old_state, new_state):
        """시장 상태 변화 알림
        
        Args:
            ticker (str): 코인 티커
            old_state (dict): 이전 상태 정보
            new_state (dict): 새 상태 정보
        """
        # 상태 변화 감지 및 방향 결정
        old_trend = old_state.get('characteristics', {}).get('trend', '')
        new_trend = new_state.get('characteristics', {}).get('trend', '')
        
        emoji = "🔄"
        if '상승' in new_trend:
            emoji = "📈"
        elif '하락' in new_trend:
            emoji = "📉"
        
        # 특성 문자열 생성
        characteristics = new_state.get('characteristics', {})
        characteristics_str = ', '.join([f"{k}: {v}" for k, v in characteristics.items()])
        
        message = (
            f"{emoji} *{ticker} 시장 상태 변화*\n"
            f"이전: {old_state.get('state_id')}\n"
            f"현재: {new_state.get('state_id')}\n"
            f"특성: {characteristics_str}"
        )
        
        self.send_message(message)
    
    def notify_model_update(self, model_info):
        """모델 업데이트 알림
        
        Args:
            model_info (dict): 모델 업데이트 정보
        """
        ticker = model_info.get('ticker', '')
        model_type = model_info.get('model_type', '')
        performance = model_info.get('performance', 0)
        improvement = model_info.get('improvement', 0)
        
        if model_type == 'ml':
            emoji = "🧠"
            message = (
                f"{emoji} *{ticker} ML 모델 재학습 완료*\n"
                f"정확도: {performance:.2f}%\n"
                f"개선도: {improvement:+.2f}%"
            )
        elif model_type == 'optimization':
            emoji = "⚙️"
            message = (
                f"{emoji} *{ticker} 전략 파라미터 최적화 완료*\n"
                f"수익률: {performance:.2f}%\n"
                f"개선도: {improvement:+.2f}%"
            )
        else:
            emoji = "🤖"
            message = (
                f"{emoji} *{ticker} 모델 업데이트*\n"
                f"성능: {performance:.2f}\n"
                f"개선도: {improvement:+.2f}%"
            )
        
        self.send_message(message)
    
    def notify_daily_summary(self):
        """일일 요약 보고서 알림"""
        # 현재 날짜
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 오늘의 거래 필터링
        today_trades = [
            trade for trade in self.daily_trades
            if trade['timestamp'].strftime("%Y-%m-%d") == today
        ]
        
        # 거래가 없는 경우
        if not today_trades:
            message = (
                f"📋 *일일 요약 ({today})*\n"
                f"오늘 거래 내역이 없습니다."
            )
            self.send_message(message)
            return
        
        # 거래 통계 계산
        buy_count = sum(1 for t in today_trades if t['action'] == 'buy')
        sell_count = sum(1 for t in today_trades if t['action'] in ['sell', 'stop_loss', 'take_profit'])
        
        # 수익/손실 거래 분석
        profit_trades = [t for t in today_trades if t['action'] in ['sell', 'take_profit'] and t['profit_percent'] > 0]
        loss_trades = [t for t in today_trades if t['action'] in ['sell', 'stop_loss'] and t['profit_percent'] <= 0]
        
        avg_profit = 0
        if profit_trades:
            avg_profit = sum(t['profit_percent'] for t in profit_trades) / len(profit_trades)
            
        avg_loss = 0
        if loss_trades:
            avg_loss = sum(t['profit_percent'] for t in loss_trades) / len(loss_trades)
        
        # 포트폴리오 정보 (일별 요약에 저장된 데이터 사용)
        portfolio_info = ""
        if self.daily_summaries:
            latest_summary = self.daily_summaries[-1]
            portfolio_info = (
                f"💼 *포트폴리오*\n"
                f"총액: {latest_summary.get('total_value', 0):,.0f}원\n"
                f"수익률: {latest_summary.get('profit_percent', 0):.2f}%\n"
            )
        
        # 최고/최저 성과 코인
        best_coin = ""
        worst_coin = ""
        if len(profit_trades) > 0:
            best_trade = max(profit_trades, key=lambda x: x['profit_percent'])
            best_coin = f"🏆 최고 성과: {best_trade['ticker']} (+{best_trade['profit_percent']:.2f}%)\n"
            
        if len(loss_trades) > 0:
            worst_trade = min(loss_trades, key=lambda x: x['profit_percent'])
            worst_coin = f"📉 최저 성과: {worst_trade['ticker']} ({worst_trade['profit_percent']:.2f}%)\n"
        
        # 메시지 구성
        message = (
            f"📋 *일일 성과 요약 ({today})*\n\n"
            f"{portfolio_info}\n"
            f"🔄 *거래 내역*\n"
            f"매수: {buy_count}건\n"
            f"매도: {sell_count}건\n\n"
            f"✅ 수익 거래: {len(profit_trades)}건 (평균: +{avg_profit:.2f}%)\n"
            f"❌ 손실 거래: {len(loss_trades)}건 (평균: {avg_loss:.2f}%)\n\n"
            f"{best_coin}"
            f"{worst_coin}"
        )
        
        self.send_message(message)
        
        # 과거 거래 정리 (지난 7일만 유지)
        week_ago = datetime.now() - timedelta(days=7)
        self.daily_trades = [
            trade for trade in self.daily_trades
            if trade['timestamp'] > week_ago
        ]
    
    def update_portfolio_summary(self, portfolio_data):
        """포트폴리오 요약 정보 업데이트
        
        Args:
            portfolio_data (dict): 포트폴리오 정보
        """
        summary = {
            'timestamp': datetime.now(),
            'total_value': portfolio_data.get('total_value', 0),
            'balance': portfolio_data.get('balance', 0),
            'profit_percent': portfolio_data.get('total_profit_percent', 0),
            'holdings': portfolio_data.get('holdings', {})
        }
        
        self.daily_summaries.append(summary)
        
        # 과거 요약 정리 (지난 30일만 유지)
        month_ago = datetime.now() - timedelta(days=30)
        self.daily_summaries = [
            s for s in self.daily_summaries
            if s['timestamp'] > month_ago
        ]
    
    def schedule_daily_report(self, time="21:00"):
        """일일 보고서 예약
        
        Args:
            time (str): 시간 (HH:MM 형식)
        """
        schedule.every().day.at(time).do(self.notify_daily_summary)
        logger.info(f"일일 요약 보고서 예약됨: 매일 {time}")
        
        # 스케줄러 스레드 시작
        if not self.running:
            self.running = True
            scheduler_thread = Thread(target=self._run_scheduler)
            scheduler_thread.daemon = True
            scheduler_thread.start()
    
    def _run_scheduler(self):
        """스케줄러 실행 스레드"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"스케줄러 오류: {e}")
                time.sleep(60)  # 오류 발생 시 1분 대기 후 재시도
    
    def stop(self):
        """알림 시스템 종료"""
        self.running = False
        logger.info("텔레그램 알림 시스템 종료")
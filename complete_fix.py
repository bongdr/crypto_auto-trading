#!/usr/bin/env python3
"""
완전한 수정 및 텔레그램 통합 스크립트
Import 오류 해결 + 텔레그램 알림 기능 추가
"""

import os

def fix_test_script():
    """테스트 스크립트의 변수 스코프 문제 수정"""
    
    test_content = '''#!/usr/bin/env python3
"""
수정된 테스트 스크립트
"""

def test_imports():
    """Import 테스트"""
    try:
        print("📦 Import 테스트 시작...")
        
        from data.improved_coin_selector import ImprovedCoinSelector
        print("✅ ImprovedCoinSelector import 성공")
        
        from strategy.improved_ml_strategy import ImprovedMLStrategy
        print("✅ ImprovedMLStrategy import 성공")
        
        from trading.risk_manager import RiskManager
        print("✅ RiskManager import 성공")
        
        from utils.system_monitor import SystemMonitor
        print("✅ SystemMonitor import 성공")
        
        print("🎉 모든 Import 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ Import 오류: {e}")
        return False

def test_basic_functionality():
    """기본 기능 테스트 - 수정된 버전"""
    try:
        print("🔧 기본 기능 테스트 시작...")
        
        # Import를 함수 내에서 다시 실행
        from data.improved_coin_selector import ImprovedCoinSelector
        from strategy.improved_ml_strategy import ImprovedMLStrategy
        from trading.risk_manager import RiskManager
        from utils.system_monitor import SystemMonitor
        
        # 코인 선택기 테스트
        coin_selector = ImprovedCoinSelector()
        tickers = coin_selector.get_krw_tickers()
        print(f"✅ 티커 조회: {len(tickers)}개")
        
        # ML 전략 테스트
        ml_strategy = ImprovedMLStrategy("KRW-BTC")
        print("✅ ML 전략 초기화 성공")
        
        # 리스크 관리자 테스트
        risk_manager = RiskManager()
        print("✅ 리스크 관리자 초기화 성공")
        
        # 시스템 모니터 테스트
        monitor = SystemMonitor()
        print("✅ 시스템 모니터 초기화 성공")
        
        print("🎉 기본 기능 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 기능 테스트 오류: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return False

def test_telegram_config():
    """텔레그램 설정 테스트"""
    try:
        print("📱 텔레그램 설정 테스트 시작...")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        import os
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if bot_token and chat_id:
            print(f"✅ 텔레그램 BOT_TOKEN: {bot_token[:10]}...")
            print(f"✅ 텔레그램 CHAT_ID: {chat_id}")
            
            # 텔레그램 알림 테스트 (실제 전송은 하지 않음)
            from utils.telegram_notifier import TelegramNotifier
            notifier = TelegramNotifier(bot_token, chat_id)
            print("✅ 텔레그램 알림 초기화 성공")
            
            return True
        else:
            print("⚠️ 텔레그램 설정이 .env에 없습니다")
            return False
            
    except Exception as e:
        print(f"❌ 텔레그램 테스트 오류: {e}")
        return False

if __name__ == "__main__":
    print("🚀 완전한 시스템 테스트 시작")
    print("="*50)
    
    # Import 테스트
    import_ok = test_imports()
    
    if import_ok:
        # 기본 기능 테스트
        func_ok = test_basic_functionality()
        
        # 텔레그램 테스트
        telegram_ok = test_telegram_config()
        
        if func_ok and telegram_ok:
            print("\\n🎉 모든 테스트 통과! 완전한 시스템 사용 준비 완료")
            print("\\n다음 명령어로 시스템을 시작하세요:")
            print("python improved_main.py")
        elif func_ok:
            print("\\n✅ 기본 기능 정상, 텔레그램 설정 확인 필요")
            print("python improved_main.py 실행 가능")
        else:
            print("\\n⚠️ 일부 기능에 문제가 있습니다.")
    else:
        print("\\n❌ Import 오류로 테스트 중단")
'''
    
    with open('test_complete_fix.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ 수정된 테스트 스크립트 생성 완료")

def create_telegram_notifier():
    """텔레그램 알림 모듈 생성"""
    
    telegram_content = '''
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
            
            message = f"{emoji} **거래 알림**\\n"
            message += f"코인: `{ticker}`\\n"
            message += f"동작: {action_text}\\n"
            message += f"가격: {price:,.0f}원\\n"
            message += f"수량: {quantity:.4f}개\\n"
            message += f"금액: {price * quantity:,.0f}원\\n"
            
            if profit is not None:
                profit_emoji = "📈" if profit > 0 else "📉"
                message += f"수익: {profit_emoji} {profit:+,.0f}원 ({profit/(price*quantity)*100:+.2f}%)\\n"
            
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
            
            message = f"📊 **일일 성과 보고서**\\n\\n"
            message += f"현재 자산: {portfolio_value:,.0f}원\\n"
            message += f"초기 투자: {initial_value:,.0f}원\\n"
            message += f"수익률: {emoji} {total_return:+.2f}%\\n"
            message += f"총 거래: {trades_count}회\\n"
            
            if summary:
                message += f"\\n**추가 지표:**\\n"
                if 'sharpe_ratio' in summary:
                    message += f"샤프 비율: {summary['sharpe_ratio']:.2f}\\n"
                if 'max_drawdown' in summary:
                    message += f"최대 손실: {summary['max_drawdown']:.2%}\\n"
                if 'volatility' in summary:
                    message += f"변동성: {summary['volatility']:.2%}\\n"
            
            message += f"\\n보고 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
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
            
            message = f"{emoji} **시스템 알림**\\n\\n"
            message += f"{message_text}\\n\\n"
            message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"시스템 알림 전송 오류: {e}")
            return False
'''
    
    os.makedirs('utils', exist_ok=True)
    with open('utils/telegram_notifier.py', 'w', encoding='utf-8') as f:
        f.write(telegram_content)
    
    print("✅ 텔레그램 알림 모듈 생성 완료")

def create_enhanced_main():
    """텔레그램 통합된 메인 스크립트 생성"""
    
    enhanced_main = '''#!/usr/bin/env python3
"""
텔레그램 통합 개선된 자동매매 시스템
"""

import os
import time
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from data.improved_coin_selector import ImprovedCoinSelector
from strategy.improved_ml_strategy import ImprovedMLStrategy
from trading.risk_manager import RiskManager
from utils.system_monitor import SystemMonitor
from utils.telegram_notifier import TelegramNotifier
from utils.logger import setup_logger

logger = setup_logger("enhanced_trading_main")

class EnhancedTradingSystem:
    """텔레그램 통합 개선된 자동매매 시스템"""
    
    def __init__(self, initial_balance=20000000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.running = False
        
        # 핵심 모듈들
        self.coin_selector = ImprovedCoinSelector()
        self.risk_manager = RiskManager()
        self.monitor = SystemMonitor()
        
        # 텔레그램 알림 초기화
        self.telegram_notifier = None
        self._setup_telegram()
        
        # 선택된 코인과 전략
        self.selected_coins = []
        self.strategies = {}
        self.positions = {}
        
        # 성능 추적
        self.trade_history = []
        self.last_rebalance = datetime.now()
        
        # 시작 알림
        if self.telegram_notifier:
            self.telegram_notifier.send_system_alert(
                'success', 
                f"🚀 자동매매 시스템 시작\\n초기 자본: {initial_balance:,}원"
            )
    
    def _setup_telegram(self):
        """텔레그램 설정"""
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if bot_token and chat_id:
                self.telegram_notifier = TelegramNotifier(bot_token, chat_id)
                logger.info("텔레그램 알림 초기화 성공")
            else:
                logger.warning("텔레그램 설정이 .env에 없습니다")
                
        except Exception as e:
            logger.error(f"텔레그램 초기화 오류: {e}")
    
    def initialize_system(self):
        """시스템 초기화"""
        try:
            logger.info("🚀 개선된 자동매매 시스템 초기화 시작")
            
            if self.telegram_notifier:
                self.telegram_notifier.send_system_alert('info', "시스템 초기화 시작...")
            
            # 1. 코인 선택
            logger.info("1️⃣ 고품질 코인 선택 중...")
            self.selected_coins, coin_scores = self.coin_selector.select_quality_coins(target_count=3)
            
            if not self.selected_coins:
                raise Exception("선택된 코인이 없습니다")
            
            logger.info(f"선택된 코인: {', '.join(self.selected_coins)}")
            
            if self.telegram_notifier:
                coins_info = "\\n".join([
                    f"• {ticker}: 점수 {coin_scores.get(ticker, {}).get('total_score', 0):.3f}"
                    for ticker in self.selected_coins
                ])
                self.telegram_notifier.send_system_alert(
                    'info',
                    f"선택된 코인 ({len(self.selected_coins)}개):\\n{coins_info}"
                )
            
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
            
            if self.telegram_notifier:
                self.telegram_notifier.send_system_alert('success', "시스템 초기화 완료!")
            
            logger.info("✅ 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.send_system_alert('error', f"초기화 실패: {str(e)}")
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
                        
                        # 성능 정보 텔레그램 전송
                        if self.telegram_notifier and hasattr(strategy, 'last_performance'):
                            perf = strategy.last_performance
                            cv_scores = perf.get('cv_scores', {})
                            if cv_scores:
                                avg_score = sum(cv_scores.values()) / len(cv_scores)
                                self.telegram_notifier.send_system_alert(
                                    'info',
                                    f"{ticker} 모델 훈련 완료\\n정확도: {avg_score:.2%}"
                                )
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
        
        if self.telegram_notifier:
            self.telegram_notifier.send_system_alert('success', "🎯 자동매매 시작!")
        
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
            if self.telegram_notifier:
                self.telegram_notifier.send_system_alert('error', f"거래 오류: {str(e)}")
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
        """거래 실행 (페이퍼 트레이딩) - 텔레그램 알림 포함"""
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
                        
                        # 텔레그램 알림
                        if self.telegram_notifier:
                            self.telegram_notifier.send_trade_alert(
                                ticker, 'buy', current_price, quantity
                            )
            
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
                
                # 텔레그램 알림
                if self.telegram_notifier:
                    self.telegram_notifier.send_trade_alert(
                        ticker, 'sell', current_price, quantity, profit
                    )
                
        except Exception as e:
            logger.error(f"{ticker} 거래 실행 오류: {e}")
    
    def _update_performance(self):
        """성능 업데이트 - 텔레그램 보고서 포함"""
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
                
                # 1시간마다 성과 보고서 텔레그램 전송 (간소화된 버전)
                current_hour = datetime.now().hour
                if current_hour in [9, 15, 21] and self.telegram_notifier:  # 하루 3번
                    self.telegram_notifier.send_performance_report(
                        portfolio_value, self.initial_balance, len(self.trade_history), summary
                    )
            
        except Exception as e:
            logger.error(f"성능 업데이트 오류: {e}")
    
    def _rebalance_portfolio(self):
        """포트폴리오 리밸런싱"""
        try:
            logger.info("🔄 포트폴리오 리밸런싱 시작")
            
            if self.telegram_notifier:
                self.telegram_notifier.send_system_alert('info', "포트폴리오 리밸런싱 시작")
            
            # 새로운 코인 선택
            new_coins, _ = self.coin_selector.select_quality_coins(target_count=3)
            
            # 기존 코인과 비교
            coins_to_remove = set(self.selected_coins) - set(new_coins)
            coins_to_add = set(new_coins) - set(self.selected_coins)
            
            if coins_to_remove or coins_to_add:
                logger.info(f"코인 변경: 제거={list(coins_to_remove)}, 추가={list(coins_to_add)}")
                
                if self.telegram_notifier:
                    change_msg = ""
                    if coins_to_remove:
                        change_msg += f"제거: {', '.join(coins_to_remove)}\\n"
                    if coins_to_add:
                        change_msg += f"추가: {', '.join(coins_to_add)}"
                    self.telegram_notifier.send_system_alert('info', f"코인 변경:\\n{change_msg}")
                
                # 제거할 코인 매도
                for ticker in coins_to_remove:
                    if self.positions[ticker]['quantity'] > 0:
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
            if self.telegram_notifier:
                self.telegram_notifier.send_system_alert('error', f"리밸런싱 오류: {str(e)}")
    
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
        
        # 최종 보고서 텔레그램 전송
        if self.telegram_notifier:
            self.telegram_notifier.send_system_alert(
                'success' if total_return > 0 else 'warning',
                f"🛑 자동매매 시스템 종료\\n\\n"
                f"초기 자본: {self.initial_balance:,}원\\n"
                f"최종 자본: {final_value:,.0f}원\\n"
                f"총 수익률: {total_return:+.2%}\\n"
                f"총 거래: {len(self.trade_history)}회"
            )
        
        logger.info("🛑 자동매매 시스템 종료")

def main():
    """메인 함수"""
    try:
        print("🚀 텔레그램 통합 자동매매 시스템 시작")
        print("="*60)
        
        # 환경 변수 확인
        required_vars = ['UPBIT_ACCESS_KEY', 'UPBIT_SECRET_KEY']
        optional_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ 필수 환경 변수가 없습니다: {', '.join(missing_vars)}")
            print("📝 .env 파일을 확인하세요")
            return
        
        # 선택적 변수 확인
        telegram_available = all(os.getenv(var) for var in optional_vars)
        if telegram_available:
            print("✅ 텔레그램 알림 사용 가능")
        else:
            print("⚠️ 텔레그램 설정이 없습니다 (선택사항)")
        
        # 시스템 초기화
        initial_balance = int(os.getenv('INITIAL_BALANCE', '20000000'))
        system = EnhancedTradingSystem(initial_balance=initial_balance)
        
        if not system.initialize_system():
            logger.error("시스템 초기화 실패")
            return
        
        print("✅ 시스템 준비 완료")
        print("🎯 자동매매 시작 (Ctrl+C로 중지)")
        
        # 거래 시작
        system.start_trading()
        
    except Exception as e:
        logger.error(f"메인 프로세스 오류: {e}")
        print(f"❌ 시스템 오류: {e}")
    finally:
        logger.info("프로그램 종료")

if __name__ == "__main__":
    main()
'''
    
    with open('enhanced_main.py', 'w', encoding='utf-8') as f:
        f.write(enhanced_main)
    
    print("✅ 텔레그램 통합 메인 스크립트 생성 완료")

def update_env_template():
    """개선된 .env 템플릿 생성"""
    
    env_template = '''# 업비트 API 키 (필수)
UPBIT_ACCESS_KEY=your_access_key_here
UPBIT_SECRET_KEY=your_secret_key_here

# 텔레그램 알림 설정 (선택사항)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# 거래 설정
TRADING_MODE=paper
INITIAL_BALANCE=20000000

# 로깅 설정
LOG_LEVEL=INFO

# 리스크 관리
STOP_LOSS_THRESHOLD=0.05
TAKE_PROFIT_THRESHOLD=0.15
MAX_POSITION_SIZE=0.3

# 데이터 품질 기준
MIN_DATA_DAYS=90
MIN_VOLUME_KRW=10000000000

# ML 모델 설정
MIN_TRAINING_SAMPLES=200
MODEL_RETRAIN_HOURS=24

# 거래 주기 (초)
TRADING_INTERVAL=3600
STATUS_CHECK_INTERVAL=1800

# 백테스팅 설정
BACKTEST_DAYS=30
'''
    
    with open('.env_enhanced_template', 'w') as f:
        f.write(env_template)
    
    print("✅ 개선된 .env 템플릿 생성 완료")

def create_requirements():
    """필요한 패키지 목록 생성"""
    
    requirements = '''pyupbit==0.2.31
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
joblib>=1.1.0
requests>=2.28.0
python-dotenv>=0.19.0
schedule>=1.1.0
'''
    
    with open('requirements_enhanced.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ 개선된 requirements.txt 생성 완료")

def main():
    """메인 실행 함수"""
    print("🔧 완전한 수정 및 텔레그램 통합 시작")
    print("="*60)
    
    # 필요한 디렉토리 생성
    directories = ['utils', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # 모든 수정 작업 실행
    fix_test_script()
    create_telegram_notifier()
    create_enhanced_main()
    update_env_template()
    create_requirements()
    
    print("\n" + "="*60)
    print("🎉 완전한 수정 및 텔레그램 통합 완료!")
    print("="*60)
    
    print("\n📱 텔레그램 알림 기능:")
    print("✅ 거래 알림 (매수/매도)")
    print("✅ 성과 보고서 (일 3회)")
    print("✅ 시스템 알림 (오류, 경고)")
    print("✅ 리밸런싱 알림")
    
    print("\n📋 다음 단계:")
    print("1. 패키지 설치: pip install -r requirements_enhanced.txt")
    print("2. 테스트 실행: python test_complete_fix.py")
    print("3. 텔레그램 통합 시스템 실행: python enhanced_main.py")
    
    print("\n⚠️ 중요:")
    print("- .env 파일에 TELEGRAM_BOT_TOKEN과 TELEGRAM_CHAT_ID가 이미 있다고 하셨으니")
    print("- 바로 텔레그램 알림 기능을 사용할 수 있습니다!")
    print("- enhanced_main.py가 텔레그램 통합 버전입니다")

if __name__ == "__main__":
    main()
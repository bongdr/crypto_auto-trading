import os
import argparse
import time
from datetime import datetime
from trading.advanced_trading_system import AdvancedTradingSystem
from utils.logger import setup_logger

logger = setup_logger("main")

def get_performance_summary(fund_manager):
    """자금 관리자 성능 요약 (안전한 접근)"""
    try:
        if not fund_manager:
            return {}
            
        summary = {
            'current_value': fund_manager.current_capital,
            'profit_ratio': (fund_manager.current_capital / fund_manager.initial_capital - 1) * 100,
            'roi': (fund_manager.current_capital / fund_manager.initial_capital - 1) * 100,
            'withdrawals': sum([w['amount'] for w in fund_manager.withdrawal_history]),
            'deposits': sum([d['amount'] for d in fund_manager.deposit_history]),
            'market_state': 'Normal'  # 기본값
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"성능 요약 생성 오류: {e}")
        return {
            'current_value': 0,
            'profit_ratio': 0,
            'roi': 0,
            'withdrawals': 0,
            'deposits': 0,
            'market_state': 'Unknown'
        }



def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='고급 가상화폐 자동 거래 시스템')
    
    parser.add_argument('--mode', type=str, default='paper', choices=['paper', 'live'], 
                        help='거래 모드 (paper: 가상거래, live: 실제거래)')
    
    parser.add_argument('--balance', type=int, default=1000000, 
                        help='초기 잔액 (원)')
    
    parser.add_argument('--coins', type=str, default='balanced', choices=['balanced', 'uncorrelated', 'basic'], 
                        help='코인 선택 방법')
    
    parser.add_argument('--coin_count', type=int, default=3, 
                        help='선택할 코인 수')
    
    parser.add_argument('--detection', type=str, default='clustering', choices=['clustering', 'hmm', 'cusum'], 
                        help='시장 상태 감지 방법')
    
    parser.add_argument('--optimization', type=str, default='genetic', choices=['bayesian', 'genetic', 'grid', 'random'], 
                        help='최적화 방법')
    
    parser.add_argument('--retraining_days', type=int, default=7, 
                        help='모델 재학습 주기 (일)')
    
    parser.add_argument('--optimization_days', type=int, default=14, 
                        help='전략 최적화 주기 (일)')
    
    parser.add_argument('--trading_interval', type=int, default=600, 
                        help='거래 확인 주기 (초)')
    
    parser.add_argument('--status_interval', type=int, default=300, 
                        help='상태 출력 주기 (초)')
    
    parser.add_argument('--log', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='로깅 레벨')
    
    parser.add_argument('--save_dir', type=str, default='results', 
                        help='결과 저장 디렉토리')
                        
    # 텔레그램 알림 관련 인수 추가
    parser.add_argument('--telegram', action='store_true',
                        help='텔레그램 알림 활성화')
                        
    parser.add_argument('--telegram_token', type=str, default='',
                        help='텔레그램 봇 토큰')
                        
    parser.add_argument('--telegram_chat_id', type=str, default='',
                        help='텔레그램 채팅 ID')
                        
    # 감정 분석 관련 인수 추가
    parser.add_argument('--sentiment', action='store_true',
                        help='감정 분석 활성화')
                        
    # 자금 관리 관련 인수 추가
    parser.add_argument('--fund_manager', action='store_true',
                        help='자동 자금 관리 활성화')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 설정 업데이트
    os.environ['TRADING_MODE'] = args.mode
    os.environ['LOG_LEVEL'] = args.log
    
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 텔레그램 설정
    telegram_config = None
    if args.telegram and args.telegram_token and args.telegram_chat_id:
        telegram_config = {
            'enabled': True,
            'token': args.telegram_token or os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'chat_id': args.telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID', ''),
            'report_time': '21:00'  # 일일 보고서 시간
        }
    
    # 시스템 초기화
    system = AdvancedTradingSystem(
        initial_balance=args.balance,
        telegram_config=telegram_config,
        enable_sentiment=args.sentiment,
        enable_fund_manager=args.fund_manager
    )
    
    # 시스템 설정
    system.detection_method = args.detection
    system.trading_interval = args.trading_interval
    system.retraining_interval = args.retraining_days
    system.optimization_interval = args.optimization_days
    
    # 코인 선택
    system.select_coins(method=args.coins, count=args.coin_count)
    
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
                if 'characteristics' in state:
                    chars = state['characteristics']
                    char_str = ', '.join([f"{k}: {v}" for k, v in chars.items()])
                    print(f"  {ticker}: {char_str}")
                else:
                    print(f"  {ticker}: {state}")
                
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
            
            # 자금 관리 상태 출력 (추가)
            if args.fund_manager and system.fund_manager:
                performance = get_performance_summary(system.fund_manager)
                print("\n=== 자금 관리 상태 ===")
                print(f"현재 포트폴리오 가치: {performance['current_value']:,.0f}원")
                print(f"수익률: {performance['profit_ratio']:.2f}%")
                print(f"ROI: {performance['roi']:.2f}%")
                print(f"총 출금액: {performance['withdrawals']:,.0f}원")
                print(f"총 입금액: {performance['deposits']:,.0f}원")
                print(f"시장 상태: {performance['market_state']}")
                print("="*60)
            
            # 상태 저장
            status_file = os.path.join(args.save_dir, f"status_{datetime.now().strftime('%Y%m%d')}.txt")
            with open(status_file, 'a') as f:
                f.write(f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                if status['portfolio']:
                    f.write(f"포트폴리오: {status['portfolio']['total_value']:,.0f}원 " +
                           f"(수익률: {status['portfolio']['total_profit_percent']:.2f}%)\n\n")
            
            time.sleep(args.status_interval)
            
    except KeyboardInterrupt:
        system.stop_trading()
        print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()
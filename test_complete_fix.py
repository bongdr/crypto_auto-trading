#!/usr/bin/env python3
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
            print("\n🎉 모든 테스트 통과! 완전한 시스템 사용 준비 완료")
            print("\n다음 명령어로 시스템을 시작하세요:")
            print("python improved_main.py")
        elif func_ok:
            print("\n✅ 기본 기능 정상, 텔레그램 설정 확인 필요")
            print("python improved_main.py 실행 가능")
        else:
            print("\n⚠️ 일부 기능에 문제가 있습니다.")
    else:
        print("\n❌ Import 오류로 테스트 중단")

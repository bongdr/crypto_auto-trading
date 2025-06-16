
#!/usr/bin/env python3
"""
빠른 수정 테스트 스크립트
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
    """기본 기능 테스트"""
    try:
        print("🔧 기본 기능 테스트 시작...")
        
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
        
        print("🎉 기본 기능 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 기능 테스트 오류: {e}")
        return False

if __name__ == "__main__":
    print("🚀 빠른 수정 테스트 시작")
    print("="*50)
    
    # Import 테스트
    import_ok = test_imports()
    
    if import_ok:
        # 기본 기능 테스트
        func_ok = test_basic_functionality()
        
        if func_ok:
            print("\n🎉 모든 테스트 통과! 시스템 사용 준비 완료")
        else:
            print("\n⚠️ 일부 기능에 문제가 있습니다.")
    else:
        print("\n❌ Import 오류로 테스트 중단")

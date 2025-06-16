#!/usr/bin/env python3
"""안정화된 메인 스크립트"""
import sys
import argparse
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='paper', choices=['paper', 'simple'])
    parser.add_argument('--balance', type=int, default=1000000)
    args = parser.parse_args()
    
    print(f"🚀 거래 시스템 시작 (모드: {args.mode})\n")
    
    try:
        if args.mode == 'simple':
            from trading.simple_trading_system import SimpleTradingSystem
            system = SimpleTradingSystem(args.balance)
        else:
            print("❌ 고급 모드는 아직 안정화 작업 중입니다")
            print("   --mode simple 을 사용하세요")
            return 1
        
        # 거래 시작
        if not system.start_trading():
            print("❌ 거래 시작 실패")
            return 1
        
        # 메인 루프
        input("\n엔터를 누르면 종료됩니다...")
        system.stop_trading()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n사용자에 의한 종료")
        return 0
    except Exception as e:
        print(f"❌ 오류: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

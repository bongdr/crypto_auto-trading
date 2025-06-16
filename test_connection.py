#!/usr/bin/env python3
"""
간단한 연결 테스트 스크립트
파일명: test_connection.py
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """필수 라이브러리 임포트 테스트"""
    try:
        import pandas as pd
        import numpy as np
        import pyupbit
        print("✅ 기본 라이브러리 임포트 성공")
        return True
    except ImportError as e:
        print(f"❌ 라이브러리 임포트 실패: {e}")
        return False

def test_upbit_connection():
    """업비트 API 연결 테스트"""
    try:
        import pyupbit
        # 공개 API 테스트 (API 키 불필요)
        tickers = pyupbit.get_tickers(fiat="KRW")
        if tickers and len(tickers) > 0:
            print(f"✅ 업비트 연결 성공 ({len(tickers)}개 코인 조회)")
            return True
        else:
            print("❌ 업비트 연결 실패: 데이터 없음")
            return False
    except Exception as e:
        print(f"❌ 업비트 연결 실패: {e}")
        return False

def test_data_collection():
    """데이터 수집 테스트"""
    try:
        import pyupbit
        # 비트코인 일봉 데이터 테스트
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=5)
        if df is not None and len(df) > 0:
            print(f"✅ 데이터 수집 성공 ({len(df)}일 데이터)")
            print(f"   최근 비트코인 가격: {df['close'].iloc[-1]:,.0f}원")
            return True
        else:
            print("❌ 데이터 수집 실패")
            return False
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        return False

if __name__ == "__main__":
    print("🔍 시스템 연결 테스트를 시작합니다...\n")
    
    tests = [
        ("라이브러리 임포트", test_imports),
        ("업비트 API 연결", test_upbit_connection), 
        ("데이터 수집", test_data_collection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"테스트 중: {test_name}")
        result = test_func()
        results.append(result)
        print()
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"🎯 테스트 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 테스트가 성공했습니다! 거래 시스템을 실행할 준비가 되었습니다.")
        print("\n다음 명령어로 페이퍼 트레이딩을 시작하세요:")
        print("python main.py --mode paper --balance 1000000")
    else:
        print("⚠️  일부 테스트가 실패했습니다. 설정을 확인해주세요.")

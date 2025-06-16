#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pyupbit

def validate_environment():
    """환경 검증"""
    print("🔍 환경 검증 중...")
    
    if not Path('.env').exists():
        print("❌ .env 파일이 없습니다")
        return False
    
    required_vars = ['TRADING_MODE', 'LOG_LEVEL']
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ {var} 환경 변수가 설정되지 않았습니다")
            return False
    
    if os.getenv('TRADING_MODE') == 'live':
        api_key = os.getenv('UPBIT_ACCESS_KEY', '')
        secret_key = os.getenv('UPBIT_SECRET_KEY', '')
        if not api_key or api_key == 'your_access_key_here':
            print("❌ 실제 거래를 위해서는 UPBIT_ACCESS_KEY가 필요합니다")
            return False
        try:
            upbit = pyupbit.Upbit(api_key, secret_key)
            upbit.get_balance("KRW")
        except:
            print("❌ API 키가 유효하지 않습니다")
            return False
    
    print("✅ 환경 검증 완료")
    return True

def validate_dependencies():
    """의존성 검증"""
    print("📦 의존성 검증 중...")
    
    required = ['pyupbit', 'pandas', 'numpy', 'scikit-learn', 'python-dotenv']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ 누락된 패키지: {', '.join(missing)}")
        print(f"   설치: pip install {' '.join(missing)}")
        return False
    
    print("✅ 의존성 검증 완료")
    return True

if __name__ == "__main__":
    print("🛡️ 시스템 검증 시작")
    
    if validate_environment() and validate_dependencies():
        print("🎉 모든 검증 통과!")
        print("   python main.py --mode paper 로 시작하세요")
        sys.exit(0)
    else:
        print("❌ 검증 실패. 문제를 해결하고 다시 실행하세요")
        sys.exit(1)

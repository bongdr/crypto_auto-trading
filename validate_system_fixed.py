#!/usr/bin/env python3
"""환경 및 보안 검증 - 수정된 버전"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def validate_environment():
    """환경 검증"""
    print("🔍 환경 검증 중...")
    
    # .env 파일 로드 (중요!)
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env 파일이 없습니다")
        return False
    
    # 환경 변수 로드
    load_dotenv()
    
    # .env 파일 내용 확인 (디버깅)
    print("📄 .env 파일 발견")
    
    # 필수 환경 변수 확인
    required_vars = ['TRADING_MODE', 'LOG_LEVEL']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            print(f"❌ {var} 환경 변수가 설정되지 않았습니다")
        else:
            print(f"✅ {var} = {value}")
    
    if missing_vars:
        print("\n💡 해결 방법:")
        print("1. .env 파일 확인: cat .env")
        print("2. 필요한 변수가 없다면 추가:")
        for var in missing_vars:
            default_value = 'paper' if var == 'TRADING_MODE' else 'INFO'
            print(f"   echo '{var}={default_value}' >> .env")
        return False
    
    # 실제 거래 모드 시 API 키 확인
    trading_mode = os.getenv('TRADING_MODE')
    if trading_mode == 'live':
        api_key = os.getenv('UPBIT_ACCESS_KEY', '')
        if not api_key or api_key == 'your_access_key_here':
            print("❌ 실제 거래를 위해서는 UPBIT_ACCESS_KEY가 필요합니다")
            return False
        else:
            print("✅ API 키 설정됨 (실제 거래 모드)")
    else:
        print(f"✅ 페이퍼 트레이딩 모드 ({trading_mode})")
    
    # 추가 환경 변수 확인
    optional_vars = ['UPBIT_ACCESS_KEY', 'UPBIT_SECRET_KEY', 'TELEGRAM_BOT_TOKEN']
    print("\n📋 선택적 환경 변수:")
    for var in optional_vars:
        value = os.getenv(var)
        if value and value != f'your_{var.lower()}_here':
            print(f"✅ {var} 설정됨")
        else:
            print(f"ℹ️  {var} 미설정 (선택사항)")
    
    print("\n✅ 환경 검증 완료")
    return True

def validate_dependencies():
    """의존성 검증"""
    print("\n📦 의존성 검증 중...")
    
    # 핵심 패키지 확인
    required = {
        'pyupbit': 'pyupbit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'dotenv': 'python-dotenv'
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
            installed.append(package_name)
        except ImportError:
            missing.append(package_name)
    
    # 결과 출력
    if installed:
        print(f"✅ 설치된 패키지: {', '.join(installed)}")
    
    if missing:
        print(f"❌ 누락된 패키지: {', '.join(missing)}")
        print(f"\n💡 설치 명령:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("✅ 의존성 검증 완료")
    return True

def check_project_structure():
    """프로젝트 구조 확인"""
    print("\n📁 프로젝트 구조 확인 중...")
    
    required_dirs = ['config', 'data', 'models', 'strategy', 'trading', 'utils', 'logs']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ 누락된 디렉토리: {', '.join(missing_dirs)}")
        print("\n💡 디렉토리 생성:")
        for dir_name in missing_dirs:
            print(f"   mkdir -p {dir_name}")
        return False
    
    print("✅ 프로젝트 구조 정상")
    return True

def check_api_connection():
    """API 연결 테스트"""
    print("\n🌐 API 연결 테스트 중...")
    
    try:
        import pyupbit
        tickers = pyupbit.get_tickers(fiat="KRW")
        if tickers and len(tickers) > 0:
            print(f"✅ 업비트 API 연결 성공 ({len(tickers)}개 종목)")
            
            # 비트코인 현재가 확인
            btc_price = pyupbit.get_current_price("KRW-BTC")
            if btc_price:
                print(f"✅ BTC 현재가: {btc_price:,.0f}원")
            return True
        else:
            print("❌ 업비트 API 응답 없음")
            return False
    except Exception as e:
        print(f"❌ API 연결 실패: {e}")
        return False

def main():
    """메인 검증 함수"""
    print("🛡️ 시스템 검증 시작")
    print("="*50)
    
    # 각 검증 수행
    checks = [
        ("환경 설정", validate_environment),
        ("의존성", validate_dependencies),
        ("프로젝트 구조", check_project_structure),
        ("API 연결", check_api_connection)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name} 검증 중 오류: {e}")
            results.append((check_name, False))
    
    # 최종 결과
    print("\n" + "="*50)
    print("📊 검증 결과:")
    print("="*50)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("\n🎉 모든 검증 통과!")
        print("   다음 명령으로 거래를 시작할 수 있습니다:")
        print("   python main.py --mode paper")
        print("   또는")
        print("   python main_stable.py --mode simple")
        return 0
    else:
        print("\n❌ 일부 검증 실패")
        print("   위의 문제를 해결하고 다시 실행하세요")
        return 1

if __name__ == "__main__":
    sys.exit(main())

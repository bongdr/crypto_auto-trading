#!/usr/bin/env python3
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_import():
    try:
        from strategy.sentiment_strategy import SentimentStrategy
        print("✅ SentimentStrategy 임포트 성공")
        
        # 인스턴스 생성 테스트
        strategy = SentimentStrategy("테스트")
        print("✅ SentimentStrategy 인스턴스 생성 성공")
        
        return True
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_import()

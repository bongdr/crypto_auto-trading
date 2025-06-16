#!/usr/bin/env python3
"""간단한 모델 검증 도구"""
import os
import joblib
from pathlib import Path

def validate_all_models():
    """모든 모델 검증"""
    model_dir = Path('saved_models')
    if not model_dir.exists():
        print("❌ saved_models 디렉토리가 없습니다")
        return
    
    models = list(model_dir.glob("*.joblib"))
    if not models:
        print("❌ 검증할 모델이 없습니다")
        return
    
    print(f"🔍 {len(models)}개 모델 검증 중...\n")
    
    for model_file in models:
        ticker = model_file.stem.replace('_ml_model', '')
        
        try:
            model_data = joblib.load(model_file)
            test_acc = model_data.get('test_accuracy', 0)
            
            if test_acc > 0.65:
                status = "✅ 우수"
            elif test_acc > 0.55:
                status = "⚠️ 보통"
            else:
                status = "❌ 개선필요"
            
            print(f"{ticker}: {test_acc:.3f} ({status})")
            
        except Exception as e:
            print(f"{ticker}: 로드 실패 - {e}")

if __name__ == "__main__":
    validate_all_models()

#!/usr/bin/env python3
import os

# 프로젝트 디렉토리 구조 생성
directories = [
    'config',
    'data',
    'models',
    'strategy',
    'trading',
    'utils',
    'logs',
    'data_cache',
    'saved_models',
    'backtest_results'
]

# 각 디렉토리 생성
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    init_file = os.path.join(directory, '__init__.py')
    if not os.path.exists(init_file) and directory not in ['logs', 'data_cache', 'saved_models', 'backtest_results']:
        with open(init_file, 'w') as f:
            f.write(f'# {directory} 패키지 초기화 파일\n')

print("프로젝트 디렉토리 구조가 생성되었습니다.")
print("다음 명령으로 필요한 패키지를 설치하세요:")
print("pip install pyupbit pandas numpy matplotlib scikit-learn joblib")
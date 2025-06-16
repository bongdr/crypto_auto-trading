# monitor_performance.py
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def monitor_performance(interval=300):
    """시스템 성능 모니터링"""
    results_dir = "results"  # 결과 저장 디렉토리
    
    plt.figure(figsize=(12, 6))
    
    while True:
        try:
            # 최신 상태 파일 찾기
            status_files = [f for f in os.listdir(results_dir) if f.startswith("status_") and f.endswith(".txt")]
            if not status_files:
                print("상태 파일을 찾을 수 없습니다.")
                time.sleep(interval)
                continue
                
            latest_file = max(status_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
            file_path = os.path.join(results_dir, latest_file)
            
            # 파일 읽기
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 수익률 데이터 추출
            timestamps = []
            profits = []
            
            for i in range(0, len(lines), 3):
                if i+1 < len(lines) and "===" in lines[i] and "포트폴리오" in lines[i+1]:
                    try:
                        timestamp = lines[i].strip().replace("=== ", "").replace(" ===", "")
                        profit_line = lines[i+1].strip()
                        profit_str = profit_line.split("수익률:")[-1].replace("%", "").strip()
                        profit = float(profit_str)
                        
                        timestamps.append(pd.to_datetime(timestamp))
                        profits.append(profit)
                    except Exception as e:
                        print(f"데이터 처리 오류: {e}")
            
            # 차트 그리기
            if timestamps and profits:
                plt.clf()
                plt.plot(timestamps, profits, marker='o')
                plt.title("가상 거래 시스템 수익률")
                plt.xlabel("시간")
                plt.ylabel("수익률 (%)")
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # 저장
                plt.savefig("performance_chart.png")
                print(f"성능 차트 업데이트: {len(profits)}개 데이터 포인트")
            else:
                print("표시할 수익률 데이터가 없습니다.")
                
        except Exception as e:
            print(f"모니터링 오류: {e}")
            
        # 대기
        time.sleep(interval)

if __name__ == "__main__":
    monitor_performance()
    
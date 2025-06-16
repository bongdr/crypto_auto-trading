import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_technical_indicators(df, ticker, save_path=None):
    """기술적 지표 시각화"""
    print(f"{ticker} 기술적 지표 차트 생성")
    
    # 간단한 차트 생성 코드
    plt.figure(figsize=(12, 8))
    
    # 가격 플롯
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['close'], label='Price')
    if 'ma5' in df.columns:
        plt.plot(df.index, df['ma5'], label='MA5')
    if 'ma20' in df.columns:
        plt.plot(df.index, df['ma20'], label='MA20')
    if 'ma60' in df.columns:
        plt.plot(df.index, df['ma60'], label='MA60')
    plt.title(f'{ticker} Price and Moving Averages')
    plt.legend()
    
    # RSI 플롯
    if 'rsi' in df.columns:
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['rsi'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-')
        plt.axhline(y=30, color='g', linestyle='-')
        plt.title('RSI')
        plt.legend()
    
    # MACD 플롯
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['macd'], label='MACD')
        plt.plot(df.index, df['macd_signal'], label='Signal')
        plt.bar(df.index, df['macd'] - df['macd_signal'], label='Histogram')
        plt.title('MACD')
        plt.legend()
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"차트 저장됨: {save_path}")
    
    plt.close()
    return True

def plot_strategy_signals(df, signals, ticker, save_path=None):
    """전략 신호 시각화"""
    print(f"{ticker} 전략 신호 차트 생성")
    
    plt.figure(figsize=(12, 6))
    
    # 가격 플롯
    plt.plot(df.index, df['close'], label='Price')
    
    # 매수/매도 신호 플롯
    buy_signals = signals == 1
    sell_signals = signals == -1
    
    plt.scatter(df.index[buy_signals], df.loc[buy_signals, 'close'], 
                marker='^', color='g', s=100, label='Buy')
    plt.scatter(df.index[sell_signals], df.loc[sell_signals, 'close'], 
                marker='v', color='r', s=100, label='Sell')
    
    plt.title(f'{ticker} Strategy Signals')
    plt.legend()
    
    # 저장
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"차트 저장됨: {save_path}")
    
    plt.close()
    return True

def plot_multi_timeframe_signals(data_dict, signals_dict, ticker, save_path=None):
    """멀티 타임프레임 신호 시각화"""
    print(f"{ticker} 멀티 타임프레임 신호 차트 생성")
    
    timeframes = list(data_dict.keys())
    n_frames = len(timeframes)
    
    plt.figure(figsize=(15, n_frames * 4))
    
    for i, tf in enumerate(timeframes):
        plt.subplot(n_frames, 1, i+1)
        
        # 데이터 및 신호
        df = data_dict[tf]
        signals = signals_dict.get(tf, pd.Series(0, index=df.index))
        
        # 가격 플롯
        plt.plot(df.index, df['close'], label=f'{tf} Price')
        
        # 매수/매도 신호 플롯
        buy_signals = signals == 1
        sell_signals = signals == -1
        
        if buy_signals.any():
            plt.scatter(df.index[buy_signals], df.loc[buy_signals, 'close'], 
                        marker='^', color='g', s=100, label='Buy')
        if sell_signals.any():
            plt.scatter(df.index[sell_signals], df.loc[sell_signals, 'close'], 
                        marker='v', color='r', s=100, label='Sell')
        
        plt.title(f'{ticker} - {tf} Timeframe')
        plt.legend()
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"차트 저장됨: {save_path}")
    
    plt.close()
    return True

def generate_strategy_report(df, signals, ticker, save_path=None):
    """전략 성과 보고서 생성"""
    print(f"{ticker} 전략 보고서 생성")
    
    # 기본 통계
    buy_count = sum(signals == 1)
    sell_count = sum(signals == -1)
    
    # 보고서 변수 초기화 - 이 줄 추가!
    report = f"=== {ticker} 전략 보고서 ===\n"
    report += f"기간: {df.index[0]} ~ {df.index[-1]}\n"
    report += f"매수 신호: {buy_count}개\n"
    report += f"매도 신호: {sell_count}개\n"
    
    # 간단한 백테스트 수행
    positions = signals.shift(1).fillna(0)  # 다음 날 포지션
    returns = df['close'].pct_change()
    strategy_returns = positions * returns
    
    # 성과 지표
    if len(strategy_returns) > 0:
        cumulative_return = (strategy_returns + 1).cumprod() - 1
        
        # 유효한 수익률 데이터가 있는 경우에만 계산
        if len(strategy_returns.dropna()) > 0:
            final_return = cumulative_return.iloc[-1]
            annual_return = ((1 + final_return) ** (252 / len(df)) - 1)
            
            # 표준편차가 0이 아닌 경우에만 샤프 비율 계산
            mean_return = strategy_returns.mean()
            std_return = strategy_returns.std()
            
            if std_return > 0:
                sharpe_ratio = mean_return / std_return * np.sqrt(252)
                report += f"누적 수익률: {final_return:.2%}\n"
                report += f"연간 수익률 (추정): {annual_return:.2%}\n"
                report += f"샤프 비율: {sharpe_ratio:.2f}\n"
            else:
                report += f"누적 수익률: {final_return:.2%}\n"
                report += f"연간 수익률 (추정): {annual_return:.2%}\n"
                report += f"샤프 비율: N/A (변동성 없음)\n"
        else:
            report += "수익률 계산을 위한 충분한 데이터가 없습니다.\n"
    else:
        report += "수익률 계산을 위한 데이터가 없습니다.\n"
    
    # NaN 처리 추가
    sharpe_ratio = 0
    if len(strategy_returns.dropna()) > 0:
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        if std_return > 0:
            sharpe_ratio = mean_return / std_return * np.sqrt(252)
        else:
            sharpe_ratio = 0
    
    # 보고서 수정
    report += f"누적 수익률: {cumulative_return.iloc[-1]:.2%}\n"
    report += f"연간 수익률 (추정): {((1 + cumulative_return.iloc[-1]) ** (252 / len(df)) - 1):.2%}\n"
    report += f"샤프 비율: {sharpe_ratio:.2f}\n"
    
    # 보고서 생성
    report = f"=== {ticker} 전략 보고서 ===\n"
    report += f"기간: {df.index[0]} ~ {df.index[-1]}\n"
    report += f"매수 신호: {buy_count}개\n"
    report += f"매도 신호: {sell_count}개\n"
    
    if len(strategy_returns) > 0:
        report += f"누적 수익률: {cumulative_return.iloc[-1]:.2%}\n"
        report += f"연간 수익률 (추정): {((1 + cumulative_return.iloc[-1]) ** (252 / len(df)) - 1):.2%}\n"
        report += f"샤프 비율: {strategy_returns.mean() / strategy_returns.std() * np.sqrt(252):.2f}\n"
    
    # 저장
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"보고서 저장됨: {save_path}")
    
    print(report)
    return report
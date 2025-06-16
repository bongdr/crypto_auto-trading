import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_technical_indicators(df, ticker, save_path=None):
    """기술적 지표 시각화"""
    print(f"{ticker} 기술적 지표 차트 생성")
    
    plt.figure(figsize=(12, 8))
    
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
    
    if 'rsi' in df.columns:
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['rsi'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-')
        plt.axhline(y=30, color='g', linestyle='-')
        plt.title('RSI')
        plt.legend()
    
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['macd'], label='MACD')
        plt.plot(df.index, df['macd_signal'], label='Signal')
        plt.bar(df.index, df['macd'] - df['macd_signal'], label='Histogram')
        plt.title('MACD')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"차트 저장됨: {save_path}")
    
    plt.close()
    return True

def plot_strategy_signals(df, signals, ticker, save_path=None):
    """전략 신호 시각화"""
    print(f"{ticker} 전략 신호 차트 생성")
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df.index, df['close'], label='Price')
    
    buy_signals = signals == 1
    sell_signals = signals == -1
    
    plt.scatter(df.index[buy_signals], df.loc[buy_signals, 'close'], 
                marker='^', color='g', s=100, label='Buy')
    plt.scatter(df.index[sell_signals], df.loc[sell_signals, 'close'], 
                marker='v', color='r', s=100, label='Sell')
    
    plt.title(f'{ticker} Strategy Signals')
    plt.legend()
    
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
        
        df = data_dict[tf]
        signals = signals_dict.get(tf, pd.Series(0, index=df.index))
        
        plt.plot(df.index, df['close'], label=f'{tf} Price')
        
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
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"차트 저장됨: {save_path}")
    
    plt.close()
    return True

def backtest_strategy(df, signals, initial_balance=1000000, commission=0.0005):
    """백테스팅 수행"""
    balance = initial_balance
    holdings = {}
    trades = []
    
    for i in range(1, len(df)):
        if signals.iloc[i] == 1 and not holdings.get('asset'):
            amount = balance * 0.99
            quantity = amount / df['close'].iloc[i]
            balance -= amount
            holdings['asset'] = {'quantity': quantity, 'price': df['close'].iloc[i]}
            trades.append({
                'timestamp': df.index[i],
                'type': 'buy',
                'price': df['close'].iloc[i],
                'quantity': quantity
            })
        elif signals.iloc[i] == -1 and holdings.get('asset'):
            quantity = holdings['asset']['quantity']
            amount = quantity * df['close'].iloc[i] * (1 - commission)
            balance += amount
            trades.append({
                'timestamp': df.index[i],
                'type': 'sell',
                'price': df['close'].iloc[i],
                'quantity': quantity
            })
            holdings.pop('asset')
    
    final_value = balance
    if holdings.get('asset'):
        final_value += holdings['asset']['quantity'] * df['close'].iloc[-1]
    
    returns = pd.Series([t['price'] / t['price'] for t in trades if t['type'] == 'sell'], index=[t['timestamp'] for t in trades if t['type'] == 'sell'])
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    return {
        'final_value': final_value,
        'return': (final_value / initial_balance - 1) * 100,
        'sharpe_ratio': sharpe_ratio,
        'trades': trades
    }

def generate_strategy_report(df, signals, ticker, save_path=None):
    """전략 성과 보고서 생성"""
    print(f"{ticker} 전략 보고서 생성")
    
    buy_count = sum(signals == 1)
    sell_count = sum(signals == -1)
    
    backtest_results = backtest_strategy(df, signals)
    
    report = f"=== {ticker} 전략 보고서 ===
"
    report += f"기간: {df.index[0]} ~ {df.index[-1]}
"
    report += f"매수 신호: {buy_count}개
"
    report += f"매도 신호: {sell_count}개
"
    report += f"최종 가치: {backtest_results['final_value']:,.0f}원
"
    report += f"수익률: {backtest_results['return']:.2f}%
"
    report += f"샤프 비율: {backtest_results['sharpe_ratio']:.2f}
"
    report += f"총 거래 횟수: {len(backtest_results['trades'])}건
"
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"보고서 저장됨: {save_path}")
    
    print(report)
    return report

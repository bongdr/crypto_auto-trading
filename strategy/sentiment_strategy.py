import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import json

from strategy.base import BaseStrategy
from sentiment.data_collector import SentimentDataCollector
from sentiment.analyzer import SentimentAnalyzer

logger = logging.getLogger("sentiment_strategy")

class SentimentStrategy(BaseStrategy):
    """감정 분석 기반 거래 전략"""
    
    def __init__(self, name="감정 분석 전략", params=None):
        super().__init__(name)
        self.params = params or {
            'sentiment_threshold': 0.2,
            'sentiment_weight': 0.4,
            'sentiment_lookback': 3,
            'extreme_sentiment_threshold': 0.7,
            'contrarian_threshold': 0.6,
            'use_contrarian': True,
            'min_news_count': 5
        }
        
        self.sentiment_collector = SentimentDataCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_cache = {}
        self.cache_dir = 'data_cache/sentiment_strategy'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"{name} 전략 초기화 완료")
    
    def _get_recent_sentiment(self, ticker, days=3):
        """최근 감정 데이터 조회"""
        cache_file = os.path.join(self.cache_dir, f"{ticker}_sentiment.json")
        
        if os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < 3600:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    logger.debug(f"{ticker} 감정 데이터 캐시에서 로드됨")
                    return cached_data
                except:
                    pass
        
        sentiment_data = self.sentiment_collector.compile_sentiment_data(ticker)
        with open(cache_file, 'w') as f:
            json.dump(sentiment_data, f)
        
        return sentiment_data
    
    def generate_sentiment_indicators(self, ticker, df):
        result_df = df.copy()
        sentiment_data = self._get_recent_sentiment(ticker)
        
        result_df['sentiment_score'] = 0
        result_df['sentiment_signal'] = 0
        result_df['sentiment_trend'] = 0
        result_df['extreme_sentiment'] = 0
        
        if not sentiment_data:
            return result_df
            
        scores = self.sentiment_analyzer.calculate_sentiment_score(sentiment_data)
        overall_score = scores.get('overall_score', 0)
        
        if scores.get('components', {}).get('news', {}).get('count', 0) >= self.params['min_news_count']:
            result_df['sentiment_score'] = overall_score
            
            if overall_score > self.params['sentiment_threshold']:
                result_df['sentiment_signal'] = 1
            elif overall_score < -self.params['sentiment_threshold']:
                result_df['sentiment_signal'] = -1
                
            if self.params['use_contrarian']:
                if overall_score > self.params['contrarian_threshold']:
                    result_df['extreme_sentiment'] = -1
                elif overall_score < -self.params['contrarian_threshold']:
                    result_df['extreme_sentiment'] = 1
        
        return result_df
    
    def generate_signal(self, df):
        ticker = df.name if hasattr(df, 'name') else 'KRW-BTC'
        df_with_sentiment = self.generate_sentiment_indicators(ticker, df)
        
        technical_signal = self._generate_technical_signal(df)
        sentiment_signal = df_with_sentiment['sentiment_signal']
        extreme_sentiment = df_with_sentiment['extreme_sentiment']
        
        combined_signal = pd.Series(0, index=df.index)
        weight_tech = 1 - self.params['sentiment_weight']
        weight_sentiment = self.params['sentiment_weight']
        
        for i in range(len(df)):
            if extreme_sentiment.iloc[i] != 0:
                combined_signal.iloc[i] = extreme_sentiment.iloc[i]
            else:
                tech_value = technical_signal.iloc[i] if i < len(technical_signal) else 0
                sent_value = sentiment_signal.iloc[i]
                weighted_signal = (tech_value * weight_tech) + (sent_value * weight_sentiment)
                
                if weighted_signal > 0.3:
                    combined_signal.iloc[i] = 1
                elif weighted_signal < -0.3:
                    combined_signal.iloc[i] = -1
        
        return combined_signal
    
    def _generate_technical_signal(self, df):
        if not all(col in df.columns for col in ['close', 'ma5', 'ma20', 'rsi']):
            return pd.Series(0, index=df.index)
        
        signal = pd.Series(0, index=df.index)
        
        try:
            ma_cross_up = (df['ma5'].shift(1) < df['ma20'].shift(1)) & (df['ma5'] > df['ma20'])
            ma_cross_down = (df['ma5'].shift(1) > df['ma20'].shift(1)) & (df['ma5'] < df['ma20'])
            rsi_oversold = df['rsi'] < 30
            rsi_overbought = df['rsi'] > 70
            
            buy_signal = ma_cross_up | rsi_oversold
            signal[buy_signal] = 1
            
            sell_signal = ma_cross_down | rsi_overbought
            signal[sell_signal] = -1
            
        except Exception as e:
            logger.error(f"기술적 신호 생성 중 오류: {e}")
        
        return signal

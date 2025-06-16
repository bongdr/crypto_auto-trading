import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger("coin_selector")

class CoinSelector:
    """가상화폐 종목 선정 도구"""
    
    def __init__(self):
        """초기화"""
        pass
        
    def get_krw_tickers(self):
        """원화 마켓 티커 목록 조회"""
        try:
            tickers = pyupbit.get_tickers(fiat="KRW")
            logger.info(f"원화 마켓 티커 {len(tickers)}개 조회됨")
            return tickers
        except Exception as e:
            logger.error(f"티커 목록 조회 오류: {e}")
            return []
            
    def get_market_info(self, tickers=None):
        """마켓 정보 조회"""
        if tickers is None:
            tickers = self.get_krw_tickers()
            
        try:
            markets = pyupbit.get_market_all()
            market_info = pd.DataFrame(markets)
            
            # 원화 마켓만 필터링
            krw_markets = market_info[market_info['market'].str.startswith('KRW-')]
            
            # 필요한 정보만 선택
            result = krw_markets[['market', 'korean_name', 'english_name']]
            result = result.set_index('market')
            
            logger.info(f"마켓 정보 {len(result)}개 조회됨")
            return result
            
        except Exception as e:
            logger.error(f"마켓 정보 조회 오류: {e}")
            return pd.DataFrame()
        
    def get_daily_candles(self, ticker, days=30):
        """일봉 데이터 조회"""
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=days)
            return df
        except Exception as e:
            logger.error(f"{ticker} 일봉 데이터 조회 오류: {e}")
            return None
            
    def calculate_metrics(self, ticker, days=30):
        """종목별 지표 계산"""
        df = self.get_daily_candles(ticker, days)
        
        if df is None or len(df) < days * 0.8:  # 데이터가 충분하지 않으면 스킵
            logger.warning(f"{ticker} 데이터 부족으로 지표 계산 실패")
            return None
            
        metrics = {}
        metrics['ticker'] = ticker
        
        # 기본 지표
        metrics['avg_price'] = df['close'].mean()
        metrics['last_price'] = df['close'].iloc[-1]
        metrics['avg_volume'] = df['volume'].mean()
        metrics['avg_volume_krw'] = (df['volume'] * df['close']).mean()
        
        # 변동성 지표
        metrics['volatility'] = df['close'].pct_change().std() * np.sqrt(252)  # 연간화된 변동성
        metrics['daily_range'] = ((df['high'] - df['low']) / df['low']).mean() * 100  # 일 평균 등락폭(%)
        
        # 추세 지표
        metrics['trend'] = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1  # 기간 내 가격 변화율
        metrics['ma5_above_ma20'] = df['close'].rolling(5).mean().iloc[-1] > df['close'].rolling(20).mean().iloc[-1]
        
        # 거래량 증가율
        metrics['volume_increase'] = (df['volume'].iloc[-5:].mean() / df['volume'].iloc[:-5].mean()) - 1
        
        # 거래량 대비 시가총액 비율 (회전율)
        # 시가총액 정보는 API에서 직접 제공하지 않아 추정치 사용
        metrics['volume_market_cap_ratio'] = metrics['avg_volume_krw'] / (metrics['avg_price'] * metrics['avg_volume'])
        
        logger.debug(f"{ticker} 지표 계산 완료")
        return metrics
    
    def select_coins(self, criteria=None, top_n=10):
        """선정 기준에 따른 코인 선택"""
        # 기본 선정 기준
        default_criteria = {
            'min_avg_volume_krw': 1000000000,  # 최소 일 평균 거래대금 (10억원)
            'min_volatility': 0.5,            # 최소 변동성
            'max_volatility': 2.0,            # 최대 변동성
            'min_daily_range': 3.0,           # 최소 일 평균 등락폭(%)
            'min_volume_increase': -0.2,      # 최소 거래량 증가율
        }
        
        if criteria:
            # 사용자 정의 기준 적용
            default_criteria.update(criteria)
            
        # 티커 목록 가져오기
        tickers = self.get_krw_tickers()
        
        # 선정 기준 로깅
        logger.info(f"코인 선정 기준: {default_criteria}")
        logger.info(f"전체 {len(tickers)}개 코인 중 선정 시작")
        
        # 각 코인별 지표 계산
        all_metrics = []
        for ticker in tickers:
            metrics = self.calculate_metrics(ticker)
            if metrics:
                all_metrics.append(metrics)
                
        if not all_metrics:
            logger.error("지표 계산에 실패했습니다")
            return []
            
        # 데이터프레임으로 변환
        metrics_df = pd.DataFrame(all_metrics)
        
        # 기준에 따른 필터링
        filtered = metrics_df[
            (metrics_df['avg_volume_krw'] >= default_criteria['min_avg_volume_krw']) &
            (metrics_df['volatility'] >= default_criteria['min_volatility']) &
            (metrics_df['volatility'] <= default_criteria['max_volatility']) &
            (metrics_df['daily_range'] >= default_criteria['min_daily_range']) &
            (metrics_df['volume_increase'] >= default_criteria['min_volume_increase'])
        ]
        
        # 결과 정렬 (거래대금 내림차순)
        filtered = filtered.sort_values('avg_volume_krw', ascending=False)
        
        # 상위 N개 선택
        selected = filtered.head(top_n)
        
        logger.info(f"선정 결과: {len(selected)}개 코인")
        for i, row in selected.iterrows():
            logger.info(f"{row['ticker']}: 평균가 {row['avg_price']:.0f}원, 변동성 {row['volatility']:.2f}, 일 평균 거래대금 {row['avg_volume_krw']/1000000000:.1f}십억원")
            
        return selected['ticker'].tolist()
        
    def select_balanced_portfolio(self, top_n=5, include_btc=True):
        """균형 잡힌 포트폴리오 선정"""
        # 시가총액 기준 상위 코인 (안정성)
        stable_coins = self.select_coins({
            'min_avg_volume_krw': 5000000000,  # 최소 일 평균 거래대금 (50억원)
            'min_volatility': 0.4,             # 최소 변동성
            'max_volatility': 1.0,             # 최대 변동성
        }, top_n=2)
        
        # 적절한 변동성을 가진 중형 코인 (균형)
        balanced_coins = self.select_coins({
            'min_avg_volume_krw': 1000000000,  # 최소 일 평균 거래대금 (10억원)
            'min_volatility': 0.7,             # 최소 변동성
            'max_volatility': 1.5,             # 최대 변동성
        }, top_n=3)
        
        # 높은 변동성을 가진 소형 코인 (기회)
        volatile_coins = self.select_coins({
            'min_avg_volume_krw': 500000000,   # 최소 일 평균 거래대금 (5억원)
            'min_volatility': 1.2,             # 최소 변동성
            'min_daily_range': 5.0,            # 최소 일 평균 등락폭(%)
        }, top_n=2)
        
        # 비트코인 추가 옵션
        portfolio = []
        if include_btc:
            portfolio.append('KRW-BTC')
            
        # 중복 제거하며 합치기
        for coin_list in [stable_coins, balanced_coins, volatile_coins]:
            for coin in coin_list:
                if coin not in portfolio:
                    portfolio.append(coin)
                    
        # 최대 코인 수 제한
        portfolio = portfolio[:top_n]
        
        logger.info(f"균형 포트폴리오 선정 결과: {len(portfolio)}개 코인")
        logger.info(f"선정 코인: {', '.join(portfolio)}")
        
        return portfolio
    
    def get_correlation_matrix(self, tickers, days=60):
        """코인 간 상관관계 분석"""
        # 각 코인의 일별 수익률 데이터 수집
        returns_data = {}
        
        for ticker in tickers:
            df = self.get_daily_candles(ticker, days)
            if df is not None and len(df) > 0:
                returns = df['close'].pct_change().dropna()
                returns_data[ticker] = returns
                
        # 데이터프레임으로 변환
        returns_df = pd.DataFrame(returns_data)
        
        # 상관관계 계산
        correlation = returns_df.corr()
        
        return correlation
        
    def select_uncorrelated_coins(self, top_n=5, max_correlation=0.5):
        """상관관계가 낮은 코인 선정"""
        # 일단 기본 기준으로 코인 선정
        candidates = self.select_coins(top_n=top_n*2)  # 후보군은 더 많이 선정
        
        if len(candidates) < 2:
            return candidates
            
        # 상관관계 계산
        corr_matrix = self.get_correlation_matrix(candidates)
        
        # 선택된 코인 목록
        selected = ['KRW-BTC']  # 비트코인은 기준 코인으로 추가
        
        # 나머지 후보 중에서 선택
        candidates.remove('KRW-BTC') if 'KRW-BTC' in candidates else None
        
        for candidate in candidates:
            # 이미 선택된 코인들과의 상관관계 확인
            is_correlated = False
            
            for selected_coin in selected:
                if selected_coin in corr_matrix.index and candidate in corr_matrix.columns:
                    correlation = corr_matrix.loc[selected_coin, candidate]
                    
                    # 상관관계가 높으면 제외
                    if abs(correlation) > max_correlation:
                        is_correlated = True
                        break
            
            # 상관관계가 낮은 경우 선택
            if not is_correlated:
                selected.append(candidate)
                
            # 목표 개수 도달 시 종료
            if len(selected) >= top_n:
                break
                
        logger.info(f"상관관계 낮은 코인 선정 결과: {len(selected)}개")
        logger.info(f"선정 코인: {', '.join(selected)}")
        
        return selected
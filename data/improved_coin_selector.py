
import logging
import numpy as np
import pandas as pd
from data.collector import UpbitDataCollector  # 수정된 import
from utils.logger import setup_logger

logger = setup_logger("improved_coin_selector")

class ImprovedCoinSelector:
    """개선된 코인 선택기 - 데이터 품질 및 안정성 중심"""
    
    def __init__(self, min_data_days=90, min_volume_krw=10_000_000_000):
        """
        Args:
            min_data_days (int): 최소 필요 데이터 일수
            min_volume_krw (int): 최소 일평균 거래대금 (원)
        """
        self.min_data_days = min_data_days
        self.min_volume_krw = min_volume_krw
        self.data_collector = UpbitDataCollector()  # 수정된 클래스명
        
    def validate_coin_data(self, ticker):
        """코인 데이터 품질 검증"""
        try:
            # 일봉 데이터 수집 - 수정된 메소드명
            df = self.data_collector.get_ohlcv(ticker, interval='day', count=120)
            
            if df is None or len(df) < self.min_data_days:
                logger.warning(f"{ticker}: 데이터 부족 ({len(df) if df is not None else 0}/{self.min_data_days}일)")
                return False
                
            # 거래량 검증 - 컬럼명 수정
            volume_col = 'volume' if 'volume' in df.columns else 'candle_acc_trade_price'
            if volume_col not in df.columns:
                logger.warning(f"{ticker}: 거래량 데이터 없음")
                return False
                
            avg_volume_krw = df[volume_col].mean()
            if avg_volume_krw < self.min_volume_krw:
                logger.warning(f"{ticker}: 거래량 부족 ({avg_volume_krw/1e9:.1f}십억원 < {self.min_volume_krw/1e9:.1f}십억원)")
                return False
                
            # 가격 안정성 검증 (급격한 변동 확인)
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # 50% 이상 변동
            if extreme_changes > 5:  # 최근 120일 중 5회 이상
                logger.warning(f"{ticker}: 가격 불안정 (극단적 변동 {extreme_changes}회)")
                return False
                
            # 연속 거래 중단 검증
            zero_volume_days = (df[volume_col] == 0).sum()
            if zero_volume_days > 3:
                logger.warning(f"{ticker}: 거래 중단일 과다 ({zero_volume_days}일)")
                return False
                
            logger.info(f"{ticker}: 데이터 품질 검증 통과")
            return True
            
        except Exception as e:
            logger.error(f"{ticker} 데이터 검증 중 오류: {e}")
            return False
    
    def calculate_coin_score(self, ticker, df):
        """코인 점수 계산 - 안정성과 수익성 균형"""
        try:
            # 거래량 컬럼 확인
            volume_col = 'volume' if 'volume' in df.columns else 'candle_acc_trade_price'
            
            # 1. 변동성 점수 (적당한 변동성 선호)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 연간 변동성
            vol_score = max(0, 1 - abs(volatility - 0.4) / 0.3)  # 40% 변동성 선호
            
            # 2. 유동성 점수
            avg_volume = df[volume_col].mean()
            liquidity_score = min(1.0, avg_volume / (50_000_000_000))  # 500억원 기준
            
            # 3. 추세 점수
            recent_trend = (df['close'].iloc[-10:].mean() / df['close'].iloc[-30:].mean()) - 1
            trend_score = max(0, min(1, (recent_trend + 0.1) / 0.2))  # -10% ~ +10% 정규화
            
            # 4. 안정성 점수
            price_stability = 1 - (df['close'].pct_change().abs() > 0.2).mean()
            
            # 가중 평균 점수
            total_score = (
                vol_score * 0.25 +
                liquidity_score * 0.35 +
                trend_score * 0.20 +
                price_stability * 0.20
            )
            
            return {
                'total_score': total_score,
                'volatility': volatility,
                'avg_volume_krw': avg_volume,
                'trend': recent_trend,
                'stability': price_stability
            }
            
        except Exception as e:
            logger.error(f"{ticker} 점수 계산 오류: {e}")
            return {'total_score': 0}
    
    def get_krw_tickers(self):
        """KRW 마켓 티커 목록 가져오기"""
        try:
            import pyupbit
            tickers = pyupbit.get_tickers(fiat="KRW")
            return tickers if tickers else []
        except Exception as e:
            logger.error(f"티커 목록 조회 오류: {e}")
            # 기본 티커 목록 반환
            return ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-MATIC', 'KRW-SOL', 'KRW-XRP']
    
    def select_quality_coins(self, target_count=3):
        """품질 기반 코인 선택"""
        logger.info("개선된 코인 선택 시작")
        
        try:
            # 전체 코인 목록 가져오기
            all_tickers = self.get_krw_tickers()
            
            # 안정적인 주요 코인들 우선 검토
            priority_coins = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-MATIC', 'KRW-SOL', 'KRW-XRP']
            other_coins = [t for t in all_tickers if t not in priority_coins]
            
            # 우선순위 코인 + 기타 코인 순으로 검토
            check_order = priority_coins + other_coins
            
            validated_coins = []
            coin_scores = {}
            
            for ticker in check_order:
                if len(validated_coins) >= target_count * 2:  # 충분한 후보 확보 시 중단
                    break
                    
                logger.info(f"{ticker} 검증 중...")
                
                if self.validate_coin_data(ticker):
                    # 데이터 재수집 및 점수 계산
                    df = self.data_collector.get_ohlcv(ticker, interval='day', count=120)
                    score_info = self.calculate_coin_score(ticker, df)
                    
                    if score_info['total_score'] > 0.3:  # 최소 점수 기준
                        validated_coins.append(ticker)
                        coin_scores[ticker] = score_info
                        logger.info(f"{ticker} 선정 완료 (점수: {score_info['total_score']:.3f})")
            
            if len(validated_coins) < target_count:
                logger.warning(f"충분한 품질의 코인을 찾지 못함 ({len(validated_coins)}/{target_count})")
                # BTC는 항상 포함
                if 'KRW-BTC' not in validated_coins:
                    validated_coins.insert(0, 'KRW-BTC')
                    df_btc = self.data_collector.get_ohlcv('KRW-BTC', interval='day', count=120)
                    if df_btc is not None:
                        coin_scores['KRW-BTC'] = self.calculate_coin_score('KRW-BTC', df_btc)
            
            # 점수 기준 정렬 및 최종 선택
            sorted_coins = sorted(validated_coins, key=lambda x: coin_scores.get(x, {}).get('total_score', 0), reverse=True)
            selected_coins = sorted_coins[:target_count]
            
            # 결과 출력
            logger.info(f"최종 선정 결과: {len(selected_coins)}개 코인")
            for ticker in selected_coins:
                score = coin_scores.get(ticker, {})
                logger.info(f"{ticker}: 점수 {score.get('total_score', 0):.3f}, "
                           f"변동성 {score.get('volatility', 0):.1%}, "
                           f"거래대금 {score.get('avg_volume_krw', 0)/1e9:.1f}십억원")
            
            return selected_coins, coin_scores
            
        except Exception as e:
            logger.error(f"코인 선택 중 오류: {e}")
            # 기본값 반환
            return ['KRW-BTC', 'KRW-ETH', 'KRW-ADA'], {}

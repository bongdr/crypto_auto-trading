# 1. data/collector.py 수정 - 데이터 수집 개선
import pandas as pd
import numpy as np
import os
import json
import time
import pyupbit
from datetime import datetime, timedelta
from utils.logger import setup_logger
from config.settings import DATA_CACHE_DIR, DATA_CACHE_EXPIRY

logger = setup_logger("data_collector")

class UpbitDataCollector:
    """업비트에서 데이터 수집 - 개선된 버전"""
    
    def __init__(self, use_cache=True):
        """초기화"""
        self.use_cache = use_cache
        self.cache_dir = DATA_CACHE_DIR
        
        # 캐시 디렉토리 생성
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_ohlcv(self, ticker, interval='day', count=200, to=None):
        """OHLCV 데이터 조회 (캐싱 적용)"""
        # 캐시 파일명 설정
        cache_key = f"{ticker}_{interval}_{count}"
        if to:
            if isinstance(to, datetime):
                to_str = to.strftime('%Y%m%d')
            else:
                to_str = to
            cache_key += f"_{to_str}"
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # 캐시 사용 시 캐시 확인
        if self.use_cache:
            # 캐시 파일 존재 및 만료 여부 확인
            if os.path.exists(cache_file):
                file_mtime = os.path.getmtime(cache_file)
                if time.time() - file_mtime < DATA_CACHE_EXPIRY:
                    try:
                        # 캐시에서 데이터 로드
                        df = pd.read_pickle(cache_file)
                        logger.debug(f"{ticker} {interval} 데이터 캐시에서 로드 (행 수: {len(df)})")
                        return df
                    except Exception as e:
                        logger.warning(f"캐시 파일 로드 실패: {e}")
        
        # 캐시가 없거나 만료된 경우 API 호출
        try:
            # API 호출
            df = pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=count, to=to)
            
            if df is None or len(df) == 0:
                logger.warning(f"{ticker} {interval} 데이터 없음")
                return None
                
            # 캐시 저장
            if self.use_cache:
                try:
                    df.to_pickle(cache_file)
                    logger.debug(f"{ticker} {interval} 데이터 캐시에 저장 (행 수: {len(df)})")
                except Exception as e:
                    logger.warning(f"캐시 파일 저장 실패: {e}")
            
            return df
            
        except ConnectionError as e:
            logger.error(f"{ticker} {interval} 데이터 조회 연결 오류: {e}")
            # 네트워크 연결 오류 시 재시도 로직
            return None
        except ValueError as e:
            logger.error(f"{ticker} {interval} 데이터 값 오류: {e}")
            return None
        except Exception as e:
            logger.error(f"{ticker} {interval} 데이터 기타 오류: {e}")
            return None
        
    def get_daily_candles(self, ticker, days=30, to=None):
        """일봉 데이터 조회"""
        return self.get_ohlcv(ticker, 'day', days, to)
    
    def get_minute_candles(self, ticker, minutes=1, count=200, to=None):
        """분봉 데이터 조회 (minutes: 1, 3, 5, 15, 30, 60, 240)"""
        if minutes not in [1, 3, 5, 15, 30, 60, 240]:
            logger.warning(f"지원하지 않는 분봉 간격: {minutes} (1, 3, 5, 15, 30, 60, 240 중 선택)")
            minutes = 1
            
        interval = f"minute{minutes}"
        return self.get_ohlcv(ticker, interval, count, to)
    
    def get_current_price(self, ticker):
        """현재가 조회"""
        try:
            price = pyupbit.get_current_price(ticker)
            logger.debug(f"{ticker} 현재가: {price}")
            return price
        except Exception as e:
            logger.error(f"{ticker} 현재가 조회 오류: {e}")
            return None
            
    def get_orderbook(self, ticker):
        """호가창 조회"""
        try:
            orderbook = pyupbit.get_orderbook(ticker)
            return orderbook
        except Exception as e:
            logger.error(f"{ticker} 호가창 조회 오류: {e}")
            return None
            
    def get_tickers(self, fiat="KRW"):
        """종목 목록 조회"""
        try:
            tickers = pyupbit.get_tickers(fiat=fiat)
            logger.debug(f"{fiat} 마켓 티커 {len(tickers)}개 조회됨")
            return tickers
        except Exception as e:
            logger.error(f"티커 목록 조회 오류: {e}")
            return []
        
    def get_market_info(self):
        """마켓 정보 조회"""
        try:
            market_info = pyupbit.get_market_all()
            return market_info
        except Exception as e:
            logger.error(f"마켓 정보 조회 오류: {e}")
            return []
            
    def collect_multiple_tickers(self, tickers, interval='day', count=30):
        """여러 종목의 데이터 수집"""
        result = {}
        
        for ticker in tickers:
            df = self.get_ohlcv(ticker, interval, count)
            if df is not None:
                result[ticker] = df
                logger.debug(f"{ticker} 데이터 수집 완료 (행 수: {len(df)})")
            else:
                logger.warning(f"{ticker} 데이터 수집 실패")
                
        logger.info(f"총 {len(result)}/{len(tickers)} 종목 데이터 수집 완료")
        return result
        
    def get_historical_data(self, ticker, days=365, interval='day'):
        """장기 과거 데이터 수집 (개선된 버전)"""
        logger.info(f"{ticker} 과거 데이터 수집 시작: {days}일, {interval}")
        
        # 캐시 확인 먼저
        cache_key = f"{ticker}_{interval}_{days}_historical"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if self.use_cache and os.path.exists(cache_file):
            file_mtime = os.path.getmtime(cache_file)
            if time.time() - file_mtime < DATA_CACHE_EXPIRY:
                try:
                    df = pd.read_pickle(cache_file)
                    if len(df) >= days * 0.8:  # 요청한 데이터의 80% 이상 있으면 사용
                        logger.info(f"{ticker} 캐시에서 {len(df)}행 로드")
                        return df
                except Exception as e:
                    logger.warning(f"캐시 로드 실패: {e}")
        
        # 실제 데이터 수집
        max_count = 200  # 한 번에 가져올 수 있는 최대 개수
        result = pd.DataFrame()
        
        # 더 넉넉하게 요청 (요청량의 120%)
        total_count = min(days * 1.2, 1000)  # 최대 1000일
        to_date = datetime.now()
        
        retry_count = 0
        max_retries = 5
        
        while total_count > 0 and retry_count < max_retries:
            try:
                # 이번 호출에서 가져올 개수
                count = min(max_count, int(total_count))
                
                logger.debug(f"{ticker} API 호출: {count}개, 종료일: {to_date.date()}")
                
                # 데이터 호출 - 여러 interval 시도
                df = None
                intervals_to_try = [interval]
                
                # interval이 day인 경우 minute240도 시도
                if interval == 'day':
                    intervals_to_try = ['day', 'minute240']
                
                for try_interval in intervals_to_try:
                    try:
                        df = pyupbit.get_ohlcv(ticker=ticker, interval=try_interval, count=count, to=to_date)
                        if df is not None and len(df) > 0:
                            logger.debug(f"{ticker} {try_interval}로 {len(df)}행 수집")
                            break
                    except Exception as e:
                        logger.warning(f"{ticker} {try_interval} 수집 실패: {e}")
                        continue
                
                if df is None or len(df) == 0:
                    logger.warning(f"{ticker} 데이터 없음, 재시도 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    time.sleep(2)  # 2초 대기 후 재시도
                    continue
                
                # 결과에 추가
                if result.empty:
                    result = df
                else:
                    # 중복 제거하며 병합
                    combined = pd.concat([df, result])
                    result = combined[~combined.index.duplicated(keep='first')].sort_index()
                
                # 다음 호출을 위한 설정
                if len(df) > 0:
                    to_date = df.index[0] - timedelta(days=1)
                    total_count -= len(df)
                else:
                    break
                
                # API 호출 제한 고려
                time.sleep(0.1)
                retry_count = 0  # 성공하면 재시도 카운트 리셋
                
            except Exception as e:
                logger.error(f"{ticker} 데이터 수집 오류: {e}")
                retry_count += 1
                time.sleep(2)
        
        # 결과 검증 및 보완
        if len(result) < days * 0.5:  # 요청한 데이터의 50% 미만이면
            logger.warning(f"{ticker} 데이터 부족: {len(result)}/{days}")
            
            # 다른 방법으로 시도 - 더 작은 단위로
            if len(result) < 50:
                logger.info(f"{ticker} 소량 데이터로 재시도")
                try:
                    df_small = pyupbit.get_ohlcv(ticker=ticker, interval='minute60', count=200)
                    if df_small is not None and len(df_small) > len(result):
                        result = df_small
                        logger.info(f"{ticker} 시간봉 데이터로 대체: {len(result)}행")
                except Exception as e:
                    logger.error(f"{ticker} 대체 데이터 수집 실패: {e}")
        
        # 최종 검증
        if len(result) > 0:
            # 캐시 저장
            if self.use_cache:
                try:
                    result.to_pickle(cache_file)
                    logger.debug(f"{ticker} 캐시 저장: {len(result)}행")
                except Exception as e:
                    logger.warning(f"캐시 저장 실패: {e}")
            
            logger.info(f"{ticker} 데이터 수집 완료: {len(result)}행")
            return result
        else:
            logger.error(f"{ticker} 데이터 수집 실패")
            return None
    
    def clear_cache(self, ticker=None, older_than_days=None):
        """캐시 삭제"""
        if not self.use_cache:
            return
            
        # 특정 티커의 캐시만 삭제
        if ticker:
            pattern = f"{ticker}_"
            count = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(pattern) and filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    
                    # 일정 기간 이상 지난 파일만 삭제하는 경우
                    if older_than_days:
                        file_mtime = os.path.getmtime(file_path)
                        file_age = (time.time() - file_mtime) / (60 * 60 * 24)  # 일 단위
                        
                        if file_age < older_than_days:
                            continue
                    
                    os.remove(file_path)
                    count += 1
            
            logger.info(f"{ticker} 캐시 파일 {count}개 삭제됨")
            
        # 모든 캐시 삭제
        else:
            count = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    
                    # 일정 기간 이상 지난 파일만 삭제하는 경우
                    if older_than_days:
                        file_mtime = os.path.getmtime(file_path)
                        file_age = (time.time() - file_mtime) / (60 * 60 * 24)  # 일 단위
                        
                        if file_age < older_than_days:
                            continue
                    
                    os.remove(file_path)
                    count += 1
            
            logger.info(f"캐시 파일 {count}개 삭제됨")
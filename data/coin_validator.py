import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("data_validator")

def validate_and_clean_data(df, min_required_days=30):
    """데이터 검증 및 정제"""
    if df is None:
        return None, "data_not_available"
    
    if len(df) < min_required_days:
        logger.warning(f"데이터 부족: {len(df)} 행 (최소 {min_required_days} 필요)")
        return None, "insufficient_data"
    
    # 결측치 확인
    missing_count = df.isnull().sum().sum()
    missing_ratio = missing_count / (df.shape[0] * df.shape[1])
    
    if missing_ratio > 0.3:  # 30% 이상 결측치면 사용 불가
        logger.warning(f"결측치 비율이 너무 높음: {missing_ratio:.1%}")
        return None, "too_many_missing"
    
    # 데이터 정제
    cleaned_df = df.copy()
    
    # 결측치 처리
    cleaned_df = cleaned_df.interpolate(method='time').ffill().bfill()
    
    # 이상치 처리
    # 거래량 0 또는 음수 값 처리
    if 'volume' in cleaned_df.columns:
        volume_median = cleaned_df['volume'].median()
        cleaned_df.loc[cleaned_df['volume'] <= 0, 'volume'] = volume_median
        
    # 가격 이상치 처리 (급격한 변동 감지)
    if 'close' in cleaned_df.columns:
        # 전일 대비 50% 이상 변동 검사
        price_change = cleaned_df['close'].pct_change().abs()
        extreme_change = price_change > 0.5
        
        if extreme_change.any():
            extreme_dates = cleaned_df.index[extreme_change]
            logger.warning(f"급격한 가격 변동 감지: {len(extreme_dates)} 행")
            
            # 이상치 데이터 보정 (전일과 다음일의 평균으로 대체)
            for date in extreme_dates:
                idx = cleaned_df.index.get_loc(date)
                if 0 < idx < len(cleaned_df) - 1:
                    prev_price = cleaned_df['close'].iloc[idx-1]
                    next_price = cleaned_df['close'].iloc[idx+1]
                    cleaned_df.loc[date, 'close'] = (prev_price + next_price) / 2
                    logger.debug(f"이상치 보정: {date}, {cleaned_df.loc[date, 'close']:.0f}원")
    
    # 중복 인덱스 제거
    cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
    
    # 무한값 처리
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
    
    # 데이터 품질 점수 계산 (0-100)
    quality_score = 100
    
    # 결측치에 따른 감점
    quality_score -= missing_ratio * 100
    
    # 데이터 길이에 따른 감점
    if len(cleaned_df) < 60:  # 60일 미만
        quality_score -= (60 - len(cleaned_df)) / 60 * 30  # 최대 30점 감점
    
    # 이상치 비율에 따른 감점
    if 'close' in cleaned_df.columns:
        extreme_ratio = extreme_change.mean() if len(extreme_change) > 0 else 0
        quality_score -= extreme_ratio * 50  # 최대 50점 감점
    
    logger.info(f"데이터 검증 완료: 품질 점수 {quality_score:.1f}/100, 행 수 {len(cleaned_df)}")
    
    return cleaned_df, "ok" if quality_score >= 70 else "low_quality"

def check_data_freshness(df, max_staleness_hours=24):
    """데이터 신선도 확인"""
    if df is None or len(df) == 0:
        return False, "no_data"
    
    # 마지막 데이터 시간
    last_time = df.index[-1]
    
    # 현재 시간과의 차이
    time_diff = pd.Timestamp.now() - last_time
    staleness_hours = time_diff.total_seconds() / 3600
    
    if staleness_hours > max_staleness_hours:
        logger.warning(f"데이터가 오래됨: {staleness_hours:.1f}시간 ({max_staleness_hours}시간 초과)")
        return False, "data_too_old"
    
    return True, "fresh_data"

def validate_ohlcv_data(df):
    """OHLCV 데이터 유효성 검사"""
    if df is None:
        return False, "data_not_available"
    
    # 필수 컬럼 확인
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"필수 컬럼 누락: {', '.join(missing_columns)}")
        return False, "missing_columns"
    
    # 데이터 일관성 검사
    inconsistent_rows = (
        (df['high'] < df['low']) | 
        (df['close'] > df['high']) | 
        (df['close'] < df['low']) | 
        (df['open'] > df['high']) | 
        (df['open'] < df['low'])
    )
    
    inconsistent_count = inconsistent_rows.sum()
    if inconsistent_count > 0:
        logger.warning(f"비일관적 OHLC 데이터: {inconsistent_count} 행")
        
        if inconsistent_count / len(df) > 0.1:  # 10% 이상 불일치
            return False, "inconsistent_data"
    
    # 거래량 검사
    if (df['volume'] <= 0).any():
        zero_volume_ratio = (df['volume'] <= 0).mean()
        logger.warning(f"거래량 0 또는 음수: {zero_volume_ratio:.1%} 행")
        
        if zero_volume_ratio > 0.2:  # 20% 이상 거래량 0
            return False, "invalid_volume"
    
    return True, "valid_data"

def detect_outliers(series, threshold=3.0):
    """이상치 감지 (Z-score 방식)"""
    if len(series) < 10:
        return pd.Series(False, index=series.index)
    
    # 이동 평균 및 표준편차 계산 (롤링 윈도우)
    rolling_mean = series.rolling(window=30, min_periods=5).mean()
    rolling_std = series.rolling(window=30, min_periods=5).std()
    
    # 기본값으로 대체 (시리즈 초반부)
    rolling_mean = rolling_mean.fillna(series.mean())
    rolling_std = rolling_std.fillna(series.std())
    
    # Z-score 계산
    z_scores = (series - rolling_mean) / rolling_std
    
    # 이상치 판별
    return abs(z_scores) > threshold

def fix_outliers(df, threshold=3.0):
    """이상치 수정"""
    fixed_df = df.copy()
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            # 이상치 감지
            outliers = detect_outliers(df[col], threshold)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"{col} 컬럼에서 {outlier_count}개 이상치 감지")
                
                # 이상치 수정 (이동 평균으로 대체)
                rolling_mean = df[col].rolling(window=5, min_periods=1).mean()
                fixed_df.loc[outliers, col] = rolling_mean.loc[outliers]
    
    return fixed_df    
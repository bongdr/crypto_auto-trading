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
    
    missing_count = df.isnull().sum().sum()
    missing_ratio = missing_count / (df.shape[0] * df.shape[1])
    
    if missing_ratio > 0.3:
        logger.warning(f"결측치 비율이 너무 높음: {missing_ratio:.1%}")
        return None, "too_many_missing"
    
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.interpolate(method='time').ffill().bfill()
    
    if 'volume' in cleaned_df.columns:
        volume_median = cleaned_df['volume'].median()
        cleaned_df.loc[cleaned_df['volume'] <= 0, 'volume'] = volume_median
        
    if 'close' in cleaned_df.columns:
        price_change = cleaned_df['close'].pct_change().abs()
        extreme_change = price_change > 0.5
        
        if extreme_change.any():
            extreme_dates = cleaned_df.index[extreme_change]
            logger.warning(f"급격한 가격 변동 감지: {len(extreme_dates)} 행")
            
            for date in extreme_dates:
                idx = cleaned_df.index.get_loc(date)
                if 0 < idx < len(cleaned_df) - 1:
                    prev_price = cleaned_df['close'].iloc[idx-1]
                    next_price = cleaned_df['close'].iloc[idx+1]
                    cleaned_df.loc[date, 'close'] = (prev_price + next_price) / 2
                    logger.debug(f"이상치 보정: {date}, {cleaned_df.loc[date, 'close']:.0f}원")
    
    cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
    
    quality_score = 100
    quality_score -= missing_ratio * 100
    
    if len(cleaned_df) < 60:
        quality_score -= (60 - len(cleaned_df)) / 60 * 30
    
    if 'close' in cleaned_df.columns:
        extreme_ratio = extreme_change.mean() if len(extreme_change) > 0 else 0
        quality_score -= extreme_ratio * 50
    
    logger.info(f"데이터 검증 완료: 품질 점수 {quality_score:.1f}/100, 행 수 {len(cleaned_df)}")
    
    return cleaned_df, "ok" if quality_score >= 70 else "low_quality"

def check_data_freshness(df, max_staleness_hours=24):
    if df is None or len(df) == 0:
        return False, "no_data"
    
    last_time = df.index[-1]
    time_diff = pd.Timestamp.now() - last_time
    staleness_hours = time_diff.total_seconds() / 3600
    
    if staleness_hours > max_staleness_hours:
        logger.warning(f"데이터가 오래됨: {staleness_hours:.1f}시간 ({max_staleness_hours}시간 초과)")
        return False, "data_too_old"
    
    return True, "fresh_data"

def validate_ohlcv_data(df):
    if df is None:
        return False, "data_not_available"
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"필수 컬럼 누락: {', '.join(missing_columns)}")
        return False, "missing_columns"
    
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
        if inconsistent_count / len(df) > 0.1:
            return False, "inconsistent_data"
    
    if (df['volume'] <= 0).any():
        zero_volume_ratio = (df['volume'] <= 0).mean()
        logger.warning(f"거래량 0 또는 음수: {zero_volume_ratio:.1%} 행")
        if zero_volume_ratio > 0.2:
            return False, "invalid_volume"
    
    return True, "valid_data"

def detect_outliers(series, method='IQR'):
    """이상치 감지 (IQR 방식)"""
    if len(series) < 10:
        return pd.Series(False, index=series.index)
    
    if method == 'IQR':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
    
    rolling_mean = series.rolling(window=30, min_periods=5).mean()
    rolling_std = series.rolling(window=30, min_periods=5).std()
    rolling_mean = rolling_mean.fillna(series.mean())
    rolling_std = rolling_std.fillna(series.std())
    z_scores = (series - rolling_mean) / rolling_std
    return abs(z_scores) > 3

def fix_outliers(df, method='IQR'):
    """이상치 수정"""
    fixed_df = df.copy()
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            outliers = detect_outliers(df[col], method=method)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"{col} 컬럼에서 {outlier_count}개 이상치 감지")
                rolling_mean = df[col].rolling(window=5, min_periods=1).mean()
                fixed_df.loc[outliers, col] = rolling_mean.loc[outliers]
    
    return fixed_df

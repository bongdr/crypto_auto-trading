
import logging
import json
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger("system_monitor")

class SystemMonitor:
    """시스템 모니터링 및 성능 추적"""
    
    def __init__(self):
        self.performance_history = []
        self.error_history = []
        self.alert_thresholds = {
            'max_drawdown': 0.1,  # 10% 최대 손실
            'consecutive_losses': 5,  # 연속 손실 횟수
            'low_accuracy': 0.4,  # ML 모델 정확도 하한
            'high_volatility': 0.8  # 포트폴리오 변동성 상한
        }
    
    def log_performance(self, portfolio_value, trades, ml_accuracy=None):
        """성능 로깅"""
        try:
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'trade_count': len(trades),
                'ml_accuracy': ml_accuracy,
                'daily_return': self._calculate_daily_return(portfolio_value)
            }
            
            self.performance_history.append(performance_data)
            
            # 최근 30일만 유지
            cutoff_date = datetime.now() - timedelta(days=30)
            self.performance_history = [
                p for p in self.performance_history 
                if datetime.fromisoformat(p['timestamp']) > cutoff_date
            ]
            
            # 알림 체크
            self._check_alerts(performance_data)
            
        except Exception as e:
            logger.error(f"성능 로깅 오류: {e}")
    
    def _calculate_daily_return(self, current_value):
        """일일 수익률 계산"""
        if len(self.performance_history) < 1:
            return 0.0
        
        yesterday_value = self.performance_history[-1]['portfolio_value']
        if yesterday_value > 0:
            return (current_value - yesterday_value) / yesterday_value
        return 0.0
    
    def _check_alerts(self, performance_data):
        """알림 조건 체크"""
        try:
            # 최대 손실률 체크
            if len(self.performance_history) >= 7:
                recent_values = [p['portfolio_value'] for p in self.performance_history[-7:]]
                max_value = max(recent_values)
                current_value = performance_data['portfolio_value']
                drawdown = (max_value - current_value) / max_value
                
                if drawdown > self.alert_thresholds['max_drawdown']:
                    logger.warning(f"⚠️ 최대 손실률 초과: {drawdown:.2%}")
            
            # ML 정확도 체크
            if performance_data.get('ml_accuracy'):
                if performance_data['ml_accuracy'] < self.alert_thresholds['low_accuracy']:
                    logger.warning(f"⚠️ ML 모델 정확도 낮음: {performance_data['ml_accuracy']:.2%}")
            
            # 연속 손실 체크
            recent_returns = [p['daily_return'] for p in self.performance_history[-5:]]
            consecutive_losses = sum(1 for r in recent_returns if r < 0)
            
            if consecutive_losses >= self.alert_thresholds['consecutive_losses']:
                logger.warning(f"⚠️ 연속 손실: {consecutive_losses}일")
                
        except Exception as e:
            logger.error(f"알림 체크 오류: {e}")
    
    def get_performance_summary(self):
        """성능 요약 반환"""
        try:
            if not self.performance_history:
                return {}
            
            recent_data = self.performance_history[-7:] if len(self.performance_history) >= 7 else self.performance_history
            
            # 기본 통계
            values = [p['portfolio_value'] for p in recent_data]
            returns = [p['daily_return'] for p in recent_data]
            
            summary = {
                'current_value': values[-1] if values else 0,
                'week_return': (values[-1] - values[0]) / values[0] if len(values) > 1 else 0,
                'avg_daily_return': sum(returns) / len(returns) if returns else 0,
                'volatility': self._calculate_volatility(returns),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(values),
                'total_trades': sum(p['trade_count'] for p in recent_data),
                'avg_ml_accuracy': self._get_avg_ml_accuracy()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"성능 요약 생성 오류: {e}")
            return {}
    
    def _calculate_volatility(self, returns):
        """변동성 계산"""
        if len(returns) < 2:
            return 0.0
        
        import numpy as np
        return np.std(returns) * np.sqrt(252)  # 연간 변동성
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        """샤프 비율 계산"""
        if len(returns) < 2:
            return 0.0
        
        import numpy as np
        excess_returns = np.array(returns) - (risk_free_rate / 252)
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
    
    def _calculate_max_drawdown(self, values):
        """최대 손실률 계산"""
        if len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _get_avg_ml_accuracy(self):
        """평균 ML 정확도"""
        accuracies = [p['ml_accuracy'] for p in self.performance_history if p.get('ml_accuracy')]
        return sum(accuracies) / len(accuracies) if accuracies else 0.0

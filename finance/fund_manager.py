import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import json

logger = logging.getLogger("fund_manager")

class FundManager:
    """자동 자금 관리 모듈"""
    
    def __init__(self, initial_capital, settings=None):
        """초기화
        
        Args:
            initial_capital (float): 초기 자본금
            settings (dict): 자금 관리 설정
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # 기본 설정
        self.settings = settings or {
            'profit_extraction_threshold': 0.2,  # 20% 수익 달성 시 일부 추출
            'profit_extraction_ratio': 0.3,      # 수익의 30% 추출
            'reinvestment_threshold': -0.1,      # -10% 손실 시 추가 투자 고려
            'reinvestment_ratio': 0.1,           # 원금의 10% 추가 투자
            'max_allocation_per_coin': 0.2,      # 한 코인당 최대 20% 할당
            'risk_adjustment_interval': 7,       # 7일마다 리스크 조정
            'emergency_fund_ratio': 0.3,         # 비상금 30% 유지
            'take_profit_thresholds': [0.1, 0.2, 0.5],  # 단계별 이익실현 임계값
            'take_profit_ratios': [0.2, 0.3, 0.4]       # 단계별 이익실현 비율
        }
        
        # 성과 관련 데이터
        self.performance_history = []  # [{timestamp, value, profit_ratio}]
        self.withdrawal_history = []   # [{timestamp, amount, reason, profit_ratio}]
        self.deposit_history = []      # [{timestamp, amount, reason, current_ratio}]
        self.profit_stages_hit = {}    # {threshold: last_hit_timestamp}
        
        # 자동 리밸런싱 설정
        self.last_rebalance = None     # 마지막 리밸런싱 날짜
        
        # 저장 디렉토리
        self.save_dir = 'data_cache/fund_manager'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 저장된 기록 로드
        self._load_history()
        
        logger.info(f"자금 관리자 초기화: 초기 자본금 {initial_capital:,.0f}원")
    
    def _load_history(self):
        """저장된 기록 로드"""
        try:
            history_file = os.path.join(self.save_dir, 'fund_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                # 날짜 문자열을 날짜 객체로 변환
                if 'performance_history' in history_data:
                    self.performance_history = history_data['performance_history']
                    for item in self.performance_history:
                        if isinstance(item['timestamp'], str):
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                
                if 'withdrawal_history' in history_data:
                    self.withdrawal_history = history_data['withdrawal_history']
                    for item in self.withdrawal_history:
                        if isinstance(item['timestamp'], str):
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                
                if 'deposit_history' in history_data:
                    self.deposit_history = history_data['deposit_history']
                    for item in self.deposit_history:
                        if isinstance(item['timestamp'], str):
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                
                if 'profit_stages_hit' in history_data:
                    self.profit_stages_hit = history_data['profit_stages_hit']
                    for threshold, timestamp in self.profit_stages_hit.items():
                        if isinstance(timestamp, str):
                            self.profit_stages_hit[threshold] = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                if 'current_capital' in history_data:
                    self.current_capital = history_data['current_capital']
                
                if 'last_rebalance' in history_data and history_data['last_rebalance']:
                    self.last_rebalance = datetime.fromisoformat(history_data['last_rebalance'].replace('Z', '+00:00'))
                
                logger.info(f"자금 관리 기록 로드 완료: {len(self.performance_history)} 항목")
        except Exception as e:
            logger.error(f"자금 관리 기록 로드 실패: {e}")
    
    def _save_history(self):
        """현재 기록 저장"""
        try:
            history_file = os.path.join(self.save_dir, 'fund_history.json')
            
            # 날짜 객체 직렬화를 위한 변환
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            
            history_data = {
                'performance_history': self.performance_history,
                'withdrawal_history': self.withdrawal_history,
                'deposit_history': self.deposit_history,
                'profit_stages_hit': self.profit_stages_hit,
                'current_capital': self.current_capital,
                'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
                'initial_capital': self.initial_capital,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, default=serialize_datetime)
                
            logger.debug("자금 관리 기록 저장 완료")
            return True
        except Exception as e:
            logger.error(f"자금 관리 기록 저장 실패: {e}")
            return False
    
    def update_portfolio_performance(self, current_value, timestamp=None):
        """포트폴리오 성과 업데이트
        
        Args:
            current_value (float): 현재 포트폴리오 가치
            timestamp (datetime): 타임스탬프 (없으면 현재 시간)
            
        Returns:
            float: 수익률
        """
        timestamp = timestamp or datetime.now()
        profit_ratio = (current_value / self.initial_capital) - 1
        
        # 성과 기록 추가
        self.performance_history.append({
            'timestamp': timestamp,
            'value': current_value,
            'profit_ratio': profit_ratio
        })
        
        # 최대 300개 항목만 유지 (약 10개월)
        if len(self.performance_history) > 300:
            self.performance_history = self.performance_history[-300:]
        
        # 현재 자본 업데이트
        self.current_capital = current_value
        
        # 기록 저장
        self._save_history()
        
        return profit_ratio
    
    def check_profit_extraction(self):
        """수익 실현 조건 확인 및 실행
        
        Returns:
            dict: 수익 실현 정보 (없으면 None)
        """
        if len(self.performance_history) < 2:
            return None
            
        # 초기 자본 대비 현재 수익률
        total_profit_ratio = (self.current_capital / self.initial_capital) - 1
        
        # 수익 실현 조건 확인
        if total_profit_ratio >= self.settings['profit_extraction_threshold']:
            # 최근 수익 실현 확인 (지난 7일 이내에 실현했으면 건너뜀)
            recent_extraction = False
            week_ago = datetime.now() - timedelta(days=7)
            
            for withdrawal in self.withdrawal_history:
                if withdrawal['reason'] == 'profit_extraction' and withdrawal['timestamp'] > week_ago:
                    recent_extraction = True
                    break
            
            if recent_extraction:
                logger.debug("최근 7일 이내에 수익을 실현했습니다")
                return None
            
            # 추출할 금액 계산
            profit_amount = self.current_capital - self.initial_capital
            extraction_amount = profit_amount * self.settings['profit_extraction_ratio']
            
            # 추출 기록
            self.withdrawal_history.append({
                'timestamp': datetime.now(),
                'amount': extraction_amount,
                'reason': 'profit_extraction',
                'profit_ratio': total_profit_ratio
            })
            
            # 자본 업데이트
            self.current_capital -= extraction_amount
            
            # 기록 저장
            self._save_history()
            
            logger.info(f"수익 실현: {extraction_amount:,.0f}원 (수익률: {total_profit_ratio:.2f})")
            
            return {
                'action': 'extract',
                'amount': extraction_amount,
                'new_capital': self.current_capital,
                'profit_ratio': total_profit_ratio
            }
            
        # 단계별 이익실현 확인
        if 'take_profit_thresholds' in self.settings and 'take_profit_ratios' in self.settings:
            for i, threshold in enumerate(self.settings['take_profit_thresholds']):
                # 해당 단계 수익률에 도달했는지 확인
                if total_profit_ratio >= threshold:
                    threshold_str = str(threshold)
                    
                    # 이미 이 단계에서 실현했는지 확인
                    if threshold_str in self.profit_stages_hit:
                        last_hit = self.profit_stages_hit[threshold_str]
                        if isinstance(last_hit, str):
                            last_hit = datetime.fromisoformat(last_hit.replace('Z', '+00:00'))
                            
                        # 30일 이내에 이미 실현했으면 건너뜀
                        if datetime.now() - last_hit < timedelta(days=30):
                            continue
                    
                    # 추출할 금액 계산
                    ratio = self.settings['take_profit_ratios'][i]
                    profit_amount = self.current_capital - self.initial_capital
                    extraction_amount = profit_amount * ratio
                    
                    # 추출 기록
                    self.withdrawal_history.append({
                        'timestamp': datetime.now(),
                        'amount': extraction_amount,
                        'reason': f'staged_profit_{threshold}',
                        'profit_ratio': total_profit_ratio
                    })
                    
                    # 자본 업데이트
                    self.current_capital -= extraction_amount
                    
                    # 단계 히트 기록
                    self.profit_stages_hit[threshold_str] = datetime.now()
                    
                    # 기록 저장
                    self._save_history()
                    
                    logger.info(f"단계별 이익실현 {threshold*100}%: {extraction_amount:,.0f}원 (수익률: {total_profit_ratio:.2f})")
                    
                    return {
                        'action': 'extract',
                        'amount': extraction_amount,
                        'new_capital': self.current_capital,
                        'profit_ratio': total_profit_ratio,
                        'stage': threshold
                    }
            
        return None
    
    def check_reinvestment(self):
        """추가 투자 조건 확인
    
        Returns:
            dict: 추가 투자 정보 (없으면 None)
        """
        if len(self.performance_history) < 2:
            return None
            
        # 현재 손실률 확인
        total_loss_ratio = (self.current_capital / self.initial_capital) - 1
        
        # 추가 투자 조건 확인
        if total_loss_ratio <= self.settings['reinvestment_threshold']:
            # 최근 추가 투자 확인 (지난 30일 이내에 투자했으면 건너뜀)
            recent_deposit = False
            month_ago = datetime.now() - timedelta(days=30)
            
            for deposit in self.deposit_history:
                if deposit['reason'] == 'reinvestment' and deposit['timestamp'] > month_ago:
                    recent_deposit = True
                    break
            
            if recent_deposit:
                logger.debug("최근 30일 이내에 추가 투자를 했습니다")
                return None
            
            # 추가 투자 금액 계산
            additional_amount = self.initial_capital * self.settings['reinvestment_ratio']
            
            # 추가 투자 기록
            self.deposit_history.append({
                'timestamp': datetime.now(),
                'amount': additional_amount,
                'reason': 'reinvestment',
                'current_ratio': total_loss_ratio
            })
            
            # 자본 업데이트
            self.current_capital += additional_amount
            
            # 기록 저장
            self._save_history()
            
            logger.info(f"추가 투자: {additional_amount:,.0f}원 (손실률: {total_loss_ratio:.2f})")
            
            return {
                'action': 'deposit',
                'amount': additional_amount,
                'new_capital': self.current_capital,
                'loss_ratio': total_loss_ratio
            }
        
        return None
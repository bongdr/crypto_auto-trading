import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import json

logger = logging.getLogger("portfolio_rebalancer")

class PortfolioRebalancer:
    """포트폴리오 주기적 리밸런싱"""
    
    def __init__(self, target_allocation=None, settings=None):
        """초기화
        
        Args:
            target_allocation (dict): 코인별 목표 할당 비율
            settings (dict): 리밸런싱 설정
        """
        self.target_allocation = target_allocation or {}  # {ticker: ratio}
        self.settings = settings or {
            'rebalance_threshold': 0.05,     # 목표 대비 5% 이상 차이 시 리밸런싱
            'rebalance_interval': 7,         # 7일마다 리밸런싱 검토
            'min_rebalance_amount': 10000,   # 최소 리밸런싱 금액 (원)
            'cash_reserve_min': 0.1,         # 최소 현금 유지 비율
            'rebalance_method': 'threshold'  # 'threshold' 또는 'periodic'
        }
        self.last_rebalance = None
        
        # 리밸런싱 기록
        self.rebalance_history = []
        
        # 저장 디렉토리
        self.save_dir = 'data_cache/rebalancer'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 저장된 기록 로드
        self._load_history()
        
        logger.info("포트폴리오 리밸런서 초기화 완료")
    
    def _load_history(self):
        """저장된 기록 로드"""
        try:
            history_file = os.path.join(self.save_dir, 'rebalance_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                # 기록 로드
                if 'rebalance_history' in history_data:
                    self.rebalance_history = history_data['rebalance_history']
                    for item in self.rebalance_history:
                        if isinstance(item['timestamp'], str):
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                
                # 타겟 할당 로드
                if 'target_allocation' in history_data:
                    self.target_allocation = history_data['target_allocation']
                
                # 마지막 리밸런싱 로드
                if 'last_rebalance' in history_data and history_data['last_rebalance']:
                    self.last_rebalance = datetime.fromisoformat(history_data['last_rebalance'].replace('Z', '+00:00'))
                
                logger.info(f"리밸런싱 기록 로드 완료: {len(self.rebalance_history)} 항목")
        except Exception as e:
            logger.error(f"리밸런싱 기록 로드 실패: {e}")
    
    def _save_history(self):
        """현재 기록 저장"""
        try:
            history_file = os.path.join(self.save_dir, 'rebalance_history.json')
            
            # 날짜 객체 직렬화를 위한 변환
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            
            history_data = {
                'rebalance_history': self.rebalance_history,
                'target_allocation': self.target_allocation,
                'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, default=serialize_datetime)
                
            logger.debug("리밸런싱 기록 저장 완료")
            return True
        except Exception as e:
            logger.error(f"리밸런싱 기록 저장 실패: {e}")
            return False
    
    def set_target_allocation(self, allocations):
        """목표 할당 비율 설정
        
        Args:
            allocations (dict): 티커별 목표 할당 비율
            
        Returns:
            bool: 설정 성공 여부
        """
        # 유효성 검사: 합이 1.0 이하인지
        if sum(allocations.values()) > 1.0:
            logger.warning(f"목표 할당 비율의 합이 1.0을 초과합니다: {sum(allocations.values())}")
            return False
            
        # 기존 목표와 비교
        changed = False
        if len(self.target_allocation) != len(allocations):
            changed = True
        else:
            for ticker, ratio in allocations.items():
                if ticker not in self.target_allocation or abs(self.target_allocation[ticker] - ratio) > 0.01:
                    changed = True
                    break
                    
        # 업데이트
        self.target_allocation = allocations.copy()
        
        # 변경 사항 저장
        if changed:
            self._save_history()
            logger.info(f"목표 할당 비율 업데이트: {allocations}")
        
        return True
    
    def check_rebalance_needed(self, current_portfolio):
        """리밸런싱 필요 여부 확인
        
        Args:
            current_portfolio (dict): 현재 코인별 평가금액
            
        Returns:
            bool: 리밸런싱 필요 여부
        """
        # 마지막 리밸런싱 후 경과 시간 확인 (주기적 리밸런싱 모드)
        if self.settings['rebalance_method'] == 'periodic' and self.last_rebalance:
            days_since_rebalance = (datetime.now() - self.last_rebalance).days
            if days_since_rebalance < self.settings['rebalance_interval']:
                logger.debug(f"마지막 리밸런싱 후 {days_since_rebalance}일 경과 (최소 {self.settings['rebalance_interval']}일)")
                return False
        
        # 목표 할당이 설정되지 않은 경우
        if not self.target_allocation:
            logger.warning("목표 할당 비율이 설정되지 않았습니다")
            return False
            
        # 현재 포트폴리오 가치가 너무 작은 경우
        total_value = sum(current_portfolio.values())
        if total_value < self.settings['min_rebalance_amount'] * 3:
            logger.debug(f"포트폴리오 가치가 너무 작습니다: {total_value:,.0f}원")
            return False
            
        # 현재 할당 비율 계산
        current_allocation = {
            ticker: value / total_value 
            for ticker, value in current_portfolio.items()
        }
        
        # 목표 할당과의 차이 확인 (임계값 기반 리밸런싱 모드)
        threshold = self.settings['rebalance_threshold']
        for ticker, target in self.target_allocation.items():
            current = current_allocation.get(ticker, 0)
            if abs(current - target) > threshold:
                logger.info(f"리밸런싱 필요: {ticker} 현재 {current:.2f} vs 목표 {target:.2f}")
                return True
                
        # 현금 비율 확인
        cash_ratio = 1 - sum(current_allocation.values())
        cash_target = 1 - sum(self.target_allocation.values())
        
        if abs(cash_ratio - cash_target) > threshold:
            logger.info(f"현금 비율 조정 필요: 현재 {cash_ratio:.2f} vs 목표 {cash_target:.2f}")
            return True
            
        logger.debug("리밸런싱 불필요: 모든 할당이 임계값 이내")
        return False
    
    def calculate_rebalance_orders(self, current_portfolio, current_prices):
        """리밸런싱 주문 계산
        
        Args:
            current_portfolio (dict): 현재 코인별 평가금액
            current_prices (dict): 코인별 현재가
            
        Returns:
            list: 리밸런싱 주문 목록
        """
        # 현재 포트폴리오 총 가치
        total_value = sum(current_portfolio.values())
        
        # 현재 할당 비율
        current_allocation = {
            ticker: value / total_value 
            for ticker, value in current_portfolio.items()
        }
        
        # 목표 금액 계산
        target_values = {
            ticker: total_value * ratio
            for ticker, ratio in self.target_allocation.items()
        }
        
        # 현금 목표
        cash_target = total_value * (1 - sum(self.target_allocation.values()))
        current_cash = total_value - sum(current_portfolio.values())
        
        # 주문 목록
        orders = []
        
        # 매도 주문 (목표보다 많이 보유 중인 코인)
        for ticker, current_value in current_portfolio.items():
            target_value = target_values.get(ticker, 0)
            
            if current_value > target_value:
                # 매도 금액
                sell_amount = current_value - target_value
                
                # 최소 주문 금액 확인
                if sell_amount < self.settings['min_rebalance_amount']:
                    continue
                    
                # 매도 주문 추가
                orders.append({
                    'ticker': ticker,
                    'type': 'sell',
                    'amount': sell_amount,
                    'price': current_prices.get(ticker, 0),
                    'target_ratio': self.target_allocation.get(ticker, 0),
                    'current_ratio': current_allocation.get(ticker, 0)
                })
        
        # 매수 주문 (목표보다 적게 보유 중인 코인)
        available_cash = current_cash
        for ticker, target_value in target_values.items():
            current_value = current_portfolio.get(ticker, 0)
            
            if current_value < target_value:
                # 매수 금액
                buy_amount = target_value - current_value
                
                # 최소 주문 금액 확인
                if buy_amount < self.settings['min_rebalance_amount']:
                    continue
                    
                # 현금 여유가 있는지 확인
                if buy_amount > available_cash:
                    # 최대한 가능한 금액으로 조정
                    buy_amount = available_cash
                    
                    # 최소 금액 이하면 건너뛰기
                    if buy_amount < self.settings['min_rebalance_amount']:
                        continue
                
                # 매수 주문 추가
                orders.append({
                    'ticker': ticker,
                    'type': 'buy',
                    'amount': buy_amount,
                    'price': current_prices.get(ticker, 0),
                    'target_ratio': self.target_allocation.get(ticker, 0),
                    'current_ratio': current_allocation.get(ticker, 0)
                })
                
                # 가용 현금 차감
                available_cash -= buy_amount
                
                # 현금이 부족하면 종료
                if available_cash < self.settings['min_rebalance_amount']:
                    break
        
        # 리밸런싱 기록 추가
        if orders:
            self.rebalance_history.append({
                'timestamp': datetime.now(),
                'total_value': total_value,
                'orders_count': len(orders),
                'orders_volume': sum(order['amount'] for order in orders),
                'tickers': [order['ticker'] for order in orders]
            })
            
            # 마지막 리밸런싱 시간 업데이트
            self.last_rebalance = datetime.now()
            
            # 기록 저장
            self._save_history()
            
            logger.info(f"리밸런싱 주문 계산 완료: {len(orders)}개 주문")
        
        return orders
    
    def execute_rebalance(self, orders, executor):
        """리밸런싱 실행
        
        Args:
            orders (list): 리밸런싱 주문 목록
            executor (TradingExecutor): 거래 실행기
            
        Returns:
            dict: 실행 결과
        """
        results = {
            'success_count': 0,
            'fail_count': 0,
            'total_volume': 0,
            'details': []
        }
        
        for order in orders:
            ticker = order['ticker']
            order_type = order['type']
            amount = order['amount']
            
            try:
                # 주문 실행
                if order_type == 'buy':
                    # 매수 주문
                    result = executor.order_manager.buy_market_order(ticker, amount)
                else:
                    # 매도 주문
                    price = order['price']
                    coin_amount = amount / price if price > 0 else 0
                    result = executor.order_manager.sell_market_order(ticker, coin_amount)
                
                # 결과 처리
                if result:
                    results['success_count'] += 1
                    results['total_volume'] += amount
                    
                    # 상세 결과 추가
                    results['details'].append({
                        'ticker': ticker,
                        'type': order_type,
                        'amount': amount,
                        'success': True
                    })
                else:
                    results['fail_count'] += 1
                    results['details'].append({
                        'ticker': ticker,
                        'type': order_type,
                        'amount': amount,
                        'success': False,
                        'reason': '주문 실패'
                    })
            except Exception as e:
                results['fail_count'] += 1
                results['details'].append({
                    'ticker': ticker,
                    'type': order_type,
                    'amount': amount,
                    'success': False,
                    'reason': str(e)
                })
        
        logger.info(f"리밸런싱 실행 결과: 성공 {results['success_count']}, 실패 {results['fail_count']}, 거래량 {results['total_volume']:,.0f}원")
        return results
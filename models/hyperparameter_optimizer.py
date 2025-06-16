import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from threading import Thread
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from functools import partial
from copy import deepcopy

from data.collector import UpbitDataCollector
from models.feature_engineering import FeatureEngineer
from trading.paper_account import PaperAccount
from utils.logger import setup_logger
from config.settings import BACKTEST_DIR



logger = setup_logger("hyperparameter_optimizer")

class HyperparameterOptimizer:
    """전략 하이퍼파라미터 자동 최적화"""
    
    def __init__(self, strategy_class, ticker="KRW-BTC", optimization_method='bayesian', parallel_jobs=4):
        """초기화"""
        self.strategy_class = strategy_class
        self.ticker = ticker
        self.optimization_method = optimization_method  # 'bayesian', 'grid', 'random', 'genetic'
        self.parallel_jobs = parallel_jobs
        
        # 데이터 수집기 및 특성 엔지니어링
        self.collector = UpbitDataCollector()
        self.fe = FeatureEngineer()
        
        # 학습 및 테스트용 데이터
        self.train_data = None
        self.test_data = None
        
        # 파라미터 공간 정의
        self.parameter_space = self._get_default_parameter_space()
        
        # 최적화 결과
        self.optimization_results = []
        
        # 베이지안 최적화 라이브러리 (옵션)
        try:
            from skopt import Optimizer
            self.bayesian_available = True
        except ImportError:
            self.bayesian_available = False
            logger.warning("Scikit-optimize 라이브러리가 없어 베이지안 최적화를 사용할 수 없습니다. pip install scikit-optimize")
        
        # 저장 경로
        self.save_dir = os.path.join(BACKTEST_DIR, "optimization_results")
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _get_default_parameter_space(self):
        """전략별 기본 파라미터 공간 정의"""
        strategy_name = self.strategy_class.__name__
        
        if strategy_name == 'TechnicalStrategy':
            return {
                'weight_ma_cross': {'type': 'float', 'min': 0.1, 'max': 2.0, 'step': 0.1},
                'weight_rsi': {'type': 'float', 'min': 0.1, 'max': 2.0, 'step': 0.1},
                'weight_bb': {'type': 'float', 'min': 0.1, 'max': 2.0, 'step': 0.1},
                'weight_macd': {'type': 'float', 'min': 0.1, 'max': 2.0, 'step': 0.1},
                'weight_stoch': {'type': 'float', 'min': 0.1, 'max': 2.0, 'step': 0.1},
                'weight_volume': {'type': 'float', 'min': 0.1, 'max': 2.0, 'step': 0.1},
                'buy_threshold': {'type': 'float', 'min': 0.1, 'max': 1.0, 'step': 0.1},
                'sell_threshold': {'type': 'float', 'min': -1.0, 'max': -0.1, 'step': 0.1},
                'rsi_overbought': {'type': 'int', 'min': 65, 'max': 85, 'step': 5},
                'rsi_oversold': {'type': 'int', 'min': 15, 'max': 35, 'step': 5}
            }
        elif strategy_name == 'MLStrategy':
            return {
                'model_type': {'type': 'categorical', 'values': ['random_forest', 'gradient_boosting']},
                'horizon': {'type': 'int', 'min': 1, 'max': 5, 'step': 1},
                'threshold': {'type': 'float', 'min': 0.005, 'max': 0.03, 'step': 0.005},
                'probability_threshold': {'type': 'float', 'min': 0.5, 'max': 0.7, 'step': 0.05}
            }
        else:
            logger.warning(f"알 수 없는 전략 유형: {strategy_name}, 기본 파라미터 공간을 사용합니다")
            return {
                'param1': {'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.1},
                'param2': {'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.1}
            }
            
    def set_parameter_space(self, param_space):
        """파라미터 공간 설정"""
        self.parameter_space = param_space
        logger.info(f"파라미터 공간 업데이트: {len(param_space)} 파라미터")
    
    def prepare_data(self, train_days=60, test_days=30):
        """최적화에 사용할 데이터 준비"""
        # 훈련 데이터 가져오기
        train_df = self.collector.get_historical_data(self.ticker, days=train_days, interval="day")
        if train_df is None or len(train_df) < train_days * 0.8:
            logger.error(f"훈련 데이터 부족: {len(train_df) if train_df is not None else 0} 행")
            return False
        
        # 테스트 데이터 가져오기
        test_df = self.collector.get_ohlcv(self.ticker, interval="day", count=test_days)
        if test_df is None or len(test_df) < test_days * 0.8:
            logger.error(f"테스트 데이터 부족: {len(test_df) if test_df is not None else 0} 행")
            return False
        
        # 특성 추가
        train_df = self.fe.add_indicators(train_df)
        test_df = self.fe.add_indicators(test_df)
        
        self.train_data = train_df
        self.test_data = test_df
        
        logger.info(f"데이터 준비 완료: 훈련 {len(train_df)} 행, 테스트 {len(test_df)} 행")
        return True
    
    def evaluate_parameters(self, params, data=None):
        """특정 파라미터 조합 평가"""
        if data is None:
            data = self.test_data
            
        if data is None:
            logger.error("평가할 데이터가 없습니다")
            return {'profit': -1.0, 'sharpe': -1.0, 'win_rate': 0.0}
        
        # 전략 인스턴스 생성
        strategy = self.strategy_class("최적화_전략", params)
        
        # 신호 생성
        signals = strategy.generate_signal(data)
        
        # 가상 계좌로 백테스트
        account = PaperAccount(initial_balance=1000000)
        current_position = 0  # 0: 없음, 1: 보유중
        trade_count = 0
        win_count = 0
        returns = []
        
        for i in range(1, len(signals)):
            signal = signals.iloc[i]
            price = data['close'].iloc[i]
            prev_price = data['close'].iloc[i-1]
            daily_return = price / prev_price - 1
            returns.append(daily_return)
            
            if signal == 1 and current_position == 0:  # 매수
                buy_amount = account.get_balance() * 0.99
                if buy_amount >= 5000:  # 최소 주문 금액
                    quantity = buy_amount / price
                    account.buy(self.ticker, price, quantity)
                    current_position = 1
                    trade_count += 0.5  # 매수는 반개 거래로 카운트
            
            elif signal == -1 and current_position == 1:  # 매도
                quantity = account.holdings.get(self.ticker, 0)
                if quantity > 0:
                    # 수익률 확인
                    avg_buy_price = account.get_avg_buy_price(self.ticker)
                    profit_percent = (price / avg_buy_price - 1) * 100
                    
                    # 승리 카운트
                    if profit_percent > 0:
                        win_count += 1
                    
                    # 매도
                    account.sell(self.ticker, price, quantity)
                    current_position = 0
                    trade_count += 0.5  # 매도는 반개 거래로 카운트 (매수-매도 쌍이 1개 거래)
                    
        # 마지막 평가
        final_balance = account.get_balance()
        final_holdings = account.get_holdings()
        final_value = final_balance
        for coin, amount in final_holdings.items():
            final_value += amount * data['close'].iloc[-1]
        
        # 수익률 계산
        profit_percent = (final_value / 1000000 - 1) * 100
        
        # 승률 계산
        win_rate = win_count / max(1, int(trade_count))
        
        # 샤프 비율 계산
        returns_array = np.array(returns)
        sharpe_ratio = 0.0
        if len(returns) > 0 and np.std(returns_array) > 0:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)  # 연간화
        
        # 결과 반환
        result = {
            'profit': profit_percent,
            'sharpe': sharpe_ratio,
            'win_rate': win_rate,
            'trade_count': int(trade_count),
            'final_value': final_value
        }
        
        return result
    
    def _sample_parameters(self, method):
        """파라미터 샘플링"""
        # 방법에 따라 파라미터 샘플링
        params = {}
        
        if method == 'random':
            # 랜덤 샘플링
            for param_name, param_config in self.parameter_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = random.uniform(param_config['min'], param_config['max'])
                elif param_config['type'] == 'int':
                    params[param_name] = random.randint(param_config['min'], param_config['max'])
                elif param_config['type'] == 'categorical':
                    params[param_name] = random.choice(param_config['values'])
        
        elif method == 'grid':
            # 그리드 포인트에서 샘플링
            for param_name, param_config in self.parameter_space.items():
                if param_config['type'] == 'float':
                    steps = int((param_config['max'] - param_config['min']) / param_config['step']) + 1
                    values = np.linspace(param_config['min'], param_config['max'], steps)
                    params[param_name] = float(random.choice(values))
                elif param_config['type'] == 'int':
                    steps = int((param_config['max'] - param_config['min']) / param_config['step']) + 1
                    values = np.linspace(param_config['min'], param_config['max'], steps, dtype=int)
                    params[param_name] = int(random.choice(values))
                elif param_config['type'] == 'categorical':
                    params[param_name] = random.choice(param_config['values'])
        
        return params
    
    def _get_parameter_ranges_for_bayesian(self):
        """베이지안 최적화를 위한 파라미터 범위 구성"""
        from skopt.space import Real, Integer, Categorical
        
        param_ranges = []
        param_names = []
        
        for param_name, param_config in self.parameter_space.items():
            param_names.append(param_name)
            
            if param_config['type'] == 'float':
                param_ranges.append(Real(param_config['min'], param_config['max']))
            elif param_config['type'] == 'int':
                param_ranges.append(Integer(param_config['min'], param_config['max']))
            elif param_config['type'] == 'categorical':
                param_ranges.append(Categorical(param_config['values']))
        
        return param_ranges, param_names            
    
    def run_optimization(self, n_trials=50, n_jobs=None):
        """최적화 실행"""
        if n_jobs is None:
            n_jobs = self.parallel_jobs
            
        if not self.train_data or not self.test_data:
            if not self.prepare_data():
                logger.error("데이터 준비 실패, 최적화를 건너뜁니다")
                return None
                
        logger.info(f"최적화 시작: 방법={self.optimization_method}, 시도={n_trials}, 병렬={n_jobs}")
        
        if self.optimization_method == 'bayesian':
            return self._run_bayesian_optimization(n_trials)
        elif self.optimization_method == 'genetic':
            return self._run_genetic_optimization(n_trials, n_jobs)
        else:
            return self._run_parallel_optimization(n_trials, n_jobs)
    
    def _run_parallel_optimization(self, n_trials, n_jobs):
        """병렬 그리드/랜덤 최적화"""
        results = []
        
        try:

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # 파라미터 샘플링 및 작업 제출
                futures = []
                for _ in range(n_trials):
                    params = self._sample_parameters(self.optimization_method)
                    future = executor.submit(self.evaluate_parameters, params, self.test_data)
                    futures.append((future, params))
                
                # 결과 수집
                for i, (future, params) in enumerate(futures):
                    try:
                        eval_result = future.result()
                        
                        # 결과 저장
                        result = {
                            'params': params,
                            'metrics': eval_result,
                            'trial': i + 1
                        }
                        results.append(result)
                        
                        logger.info(f"시도 {i+1}/{n_trials}: 수익률 {eval_result['profit']:.2f}%, 승률 {eval_result['win_rate']:.2f}")
                        
                    except Exception as e:
                        logger.error(f"평가 오류: {e}")
        except Exception as e:
            logger.error(f"병렬 처리 오류: {e}")
            logger.info("단일 스레드로 실행합니다.")
            # 단일 스레드 대체 실행
            for i in range(n_trials):
                params = self._sample_parameters(self.optimization_method)
                eval_result = self.evaluate_parameters(params, self.test_data)
                result = {
                    'params': params,
                    'metrics': eval_result,
                    'trial': i + 1
                }
                results.append(result)
                logger.info(f"시도 {i+1}/{n_trials}: 수익률 {eval_result['profit']:.2f}%, 승률 {eval_result['win_rate']:.2f}")
        
        
        # 결과 정렬 (수익률 기준)
        results.sort(key=lambda x: x['metrics']['profit'], reverse=True)
        
        # 최적화 결과 저장
        self.optimization_results = results
        
        # 최적 파라미터 반환
        if results:
            best_result = results[0]
            logger.info(f"최적 파라미터 발견: 수익률 {best_result['metrics']['profit']:.2f}%, 승률 {best_result['metrics']['win_rate']:.2f}")
            
            # 결과 저장
            self._save_optimization_results()
            
            return best_result['params']
        
        return None
    
    def _run_bayesian_optimization(self, n_trials):
        """베이지안 최적화"""
        if not self.bayesian_available:
            logger.warning("베이지안 최적화 라이브러리 없음, 랜덤 최적화로 대체합니다")
            self.optimization_method = 'random'
            return self._run_parallel_optimization(n_trials, self.parallel_jobs)
            
        from skopt import Optimizer
        from skopt.utils import use_named_args
        
        # 파라미터 범위 준비
        param_ranges, param_names = self._get_parameter_ranges_for_bayesian()
        
        # 목적 함수 정의
        @use_named_args(param_ranges)
        def objective(**params):
            eval_result = self.evaluate_parameters(params, self.test_data)
            # 음수 수익률 반환 (최소화 문제로 변환)
            return -eval_result['profit']
        
        # 옵티마이저 초기화
        optimizer = Optimizer(
            dimensions=param_ranges,
            base_estimator="GP",  # 가우시안 프로세스
            acq_func="EI",  # 기대 개선도
            n_initial_points=10  # 초기 랜덤 탐색 횟수
        )
        
        results = []
        
        # 최적화 실행
        for i in range(n_trials):
            try:
                # 다음 파라미터 조합 제안
                next_point = optimizer.ask()
                
                # 목적 함수 평가
                value = objective(next_point)
                
                # 결과 업데이트
                optimizer.tell(next_point, value)
                
                # 파라미터 매핑
                params = {param_names[j]: next_point[j] for j in range(len(param_names))}
                
                # 평가 결과 계산
                eval_result = self.evaluate_parameters(params, self.test_data)
                
                # 결과 저장
                result = {
                    'params': params,
                    'metrics': eval_result,
                    'trial': i + 1
                }
                results.append(result)
                
                logger.info(f"시도 {i+1}/{n_trials}: 수익률 {eval_result['profit']:.2f}%, 승률 {eval_result['win_rate']:.2f}")
                
            except Exception as e:
                logger.error(f"베이지안 최적화 오류: {e}")
        
        # 결과 정렬
        results.sort(key=lambda x: x['metrics']['profit'], reverse=True)
        
        # 최적화 결과 저장
        self.optimization_results = results
        
        # 최적 파라미터 반환
        if results:
            best_result = results[0]
            logger.info(f"최적 파라미터 발견: 수익률 {best_result['metrics']['profit']:.2f}%, 승률 {best_result['metrics']['win_rate']:.2f}")
            
            # 결과 저장
            self._save_optimization_results()
            
            return best_result['params']
        
        return None
    
    def _run_genetic_optimization(self, n_trials, n_jobs):
        """유전 알고리즘 기반 최적화"""
        # 유전 알고리즘 파라미터
        population_size = min(20, n_trials // 2)  # 개체 수
        generations = n_trials // population_size  # 세대 수
        mutation_rate = 0.2  # 변이율
        elite_size = max(1, population_size // 5)  # 엘리트 개체 수
        
        logger.info(f"유전 알고리즘 최적화: 개체수={population_size}, 세대수={generations}, 변이율={mutation_rate}")
        
        # 이전 최적화 결과 확인 (복구 목적)
        checkpoint_path = os.path.join(self.save_dir, f"{self.strategy_class.__name__}_{self.ticker}_genetic_checkpoint.joblib")
        start_generation = 0
        best_population = []
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = joblib.load(checkpoint_path)
                start_generation = checkpoint.get('generation', 0)
                best_population = checkpoint.get('best_population', [])
                all_results = checkpoint.get('all_results', [])
                
                logger.info(f"체크포인트 로드: 세대 {start_generation}, 결과 {len(all_results)}개")
                
                if start_generation >= generations:
                    logger.info("이미 최적화가 완료되었습니다.")
                    return all_results[0]['params'] if all_results else None
            except Exception as e:
                logger.error(f"체크포인트 로드 실패: {e}")
                start_generation = 0
                best_population = []
                all_results = []
        else:
            all_results = []
        
        # 초기 개체군 생성 (또는 이전 세대의 최고 개체 사용)
        population = best_population
        remaining_size = population_size - len(population)
        
        if remaining_size > 0:
            for _ in range(remaining_size):
                params = self._sample_parameters('random')
                population.append(params)
        
        # 세대 반복
        for generation in range(start_generation, generations):
            logger.info(f"세대 {generation+1}/{generations} 평가 중...")
            
            # 안전한 개체 평가를 위한 함수
            def safe_evaluate(params):
                try:
                    return params, self.evaluate_parameters(params, self.test_data)
                except Exception as e:
                    logger.error(f"개체 평가 오류: {e}")
                    # 기본 낮은 성능 값 반환
                    return params, {'profit': -100, 'sharpe': -1, 'win_rate': 0}
            
            try:
                # 개체 평가
                evaluated_population = []
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [executor.submit(safe_evaluate, params) for params in population]
                    
                    for future in as_completed(futures):
                        try:
                            params, eval_result = future.result()
                            evaluated_population.append((params, eval_result))
                            
                            # 결과 저장
                            trial_num = generation * population_size + len(evaluated_population)
                            result = {
                                'params': params,
                                'metrics': eval_result,
                                'trial': trial_num,
                                'generation': generation + 1
                            }
                            all_results.append(result)
                            
                            logger.info(f"개체 {len(evaluated_population)}/{population_size}: 수익률 {eval_result['profit']:.2f}%, 승률 {eval_result['win_rate']:.2f}")
                            
                        except Exception as e:
                            logger.error(f"결과 처리 오류: {e}")
                
                # 적합도 기준 정렬
                evaluated_population.sort(key=lambda x: x[1]['profit'], reverse=True)
                
                # 체크포인트 저장
                checkpoint = {
                    'generation': generation + 1,
                    'best_population': [p[0] for p in evaluated_population[:elite_size]],
                    'all_results': sorted(all_results, key=lambda x: x['metrics']['profit'], reverse=True)
                }
                joblib.dump(checkpoint, checkpoint_path)
                
                # 엘리트 선정
                elites = [p[0] for p in evaluated_population[:elite_size]]
                
                # 현재 세대 최고 결과 출력
                best_in_gen = evaluated_population[0]
                logger.info(f"세대 {generation+1} 최고 개체: 수익률 {best_in_gen[1]['profit']:.2f}%, 승률 {best_in_gen[1]['win_rate']:.2f}")
                
                # 마지막 세대면 종료
                if generation == generations - 1:
                    break
                
                # 룰렛 휠 선택
                fitness_values = [max(0, result['profit'] + 100) for _, result in evaluated_population]  # 음수 수익률 방지
                total_fitness = sum(fitness_values)
                
                if total_fitness == 0:
                    # 모든 개체가 손실인 경우
                    selection_probs = [1/len(fitness_values)] * len(fitness_values)
                else:
                    selection_probs = [fitness/total_fitness for fitness in fitness_values]
                
                # 다음 세대 생성
                next_population = elites.copy()  # 엘리트는 그대로 유지
                
                # 교배를 통한 나머지 개체 생성
                while len(next_population) < population_size:
                    try:
                        # 부모 선택
                        parent1_idx = random.choices(range(len(evaluated_population)), weights=selection_probs)[0]
                        parent2_idx = random.choices(range(len(evaluated_population)), weights=selection_probs)[0]
                        
                        parent1 = evaluated_population[parent1_idx][0]
                        parent2 = evaluated_population[parent2_idx][0]
                        
                        # 교배
                        child = {}
                        for param_name in parent1.keys():
                            # 랜덤하게 부모 중 하나의 유전자 선택
                            if random.random() < 0.5:
                                child[param_name] = parent1[param_name]
                            else:
                                child[param_name] = parent2[param_name]
                        
                        # 변이
                        if random.random() < mutation_rate:
                            # 랜덤 파라미터 선택
                            param_to_mutate = random.choice(list(child.keys()))
                            param_config = self.parameter_space[param_to_mutate]
                            
                            # 변이 적용
                            if param_config['type'] == 'float':
                                mutation_range = (param_config['max'] - param_config['min']) * 0.2  # 20% 범위 내 변이
                                new_value = child[param_to_mutate] + random.uniform(-mutation_range, mutation_range)
                                child[param_to_mutate] = max(param_config['min'], min(param_config['max'], new_value))
                            elif param_config['type'] == 'int':
                                mutation_range = max(1, int((param_config['max'] - param_config['min']) * 0.2))
                                new_value = child[param_to_mutate] + random.randint(-mutation_range, mutation_range)
                                child[param_to_mutate] = max(param_config['min'], min(param_config['max'], new_value))
                            elif param_config['type'] == 'categorical':
                                child[param_to_mutate] = random.choice(param_config['values'])
                        
                        next_population.append(child)
                    except Exception as e:
                        logger.error(f"다음 세대 생성 오류: {e}")
                        # 오류 발생 시 임의의 개체 추가
                        next_population.append(self._sample_parameters('random'))
                
                # 다음 세대로 업데이트
                population = next_population
                
            except Exception as e:
                logger.error(f"세대 {generation+1} 처리 중 오류: {e}")
                # 심각한 오류 발생 시에도 지금까지의 최적 결과 저장
                break
        
        # 최종 결과 정리
        all_results.sort(key=lambda x: x['metrics']['profit'], reverse=True)
        
        # 최적화 결과 저장
        self.optimization_results = all_results
        
        # 최적 파라미터 반환
        if all_results:
            best_result = all_results[0]
            logger.info(f"최적 파라미터 발견: 수익률 {best_result['metrics']['profit']:.2f}%, 승률 {best_result['metrics']['win_rate']:.2f}")
            
            # 결과 저장
            self._save_optimization_results()
            
            # 체크포인트 삭제 (최적화 완료)
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except:
                    pass
            
            return best_result['params']
        
        return None           
    
    def _save_optimization_results(self):
        """최적화 결과 저장"""
        if not self.optimization_results:
            logger.warning("저장할 최적화 결과가 없습니다")
            return False
            
        try:
            # 저장 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy_class.__name__
            filename = f"{strategy_name}_{self.ticker}_{self.optimization_method}_{timestamp}.json"
            filepath = os.path.join(self.save_dir, filename)
            
            # 저장 데이터 준비
            save_data = {
                'strategy': strategy_name,
                'ticker': self.ticker,
                'method': self.optimization_method,
                'timestamp': timestamp,
                'parameter_space': self.parameter_space,
                'results': self.optimization_results,
                'best_params': self.optimization_results[0]['params'],
                'best_metrics': self.optimization_results[0]['metrics']
            }
            
            # JSON 저장
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
                
            logger.info(f"최적화 결과 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"최적화 결과 저장 실패: {e}")
            return False
    
    def load_best_params(self, strategy_name=None, ticker=None):
        """가장 최근의 최적 파라미터 로드"""
        if strategy_name is None:
            strategy_name = self.strategy_class.__name__
            
        if ticker is None:
            ticker = self.ticker
            
        try:
            # 파일 목록 가져오기
            files = [f for f in os.listdir(self.save_dir) if f.startswith(f"{strategy_name}_{ticker}_") and f.endswith('.json')]
            
            if not files:
                logger.warning(f"{strategy_name} {ticker}에 대한 저장된 파라미터가 없습니다")
                return None
                
            # 가장 최근 파일 선택
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(self.save_dir, f)))
            filepath = os.path.join(self.save_dir, latest_file)
            
            # 로드
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            logger.info(f"최적 파라미터 로드 완료: {filepath}")
            
            # 결과 반환
            return data.get('best_params')
            
        except Exception as e:
            logger.error(f"최적 파라미터 로드 실패: {e}")
            return None
    
    def apply_best_params(self, strategy_instance=None):
        """최적의 파라미터를 전략에 적용"""
        # 최적 파라미터 로드
        best_params = self.load_best_params()
        
        if best_params is None:
            logger.warning("적용할 최적 파라미터가 없습니다")
            return False
            
        # 전략 인스턴스가 제공되지 않은 경우 새로 생성
        if strategy_instance is None:
            strategy_instance = self.strategy_class(f"{self.ticker}_최적화됨", best_params)
            return strategy_instance
            
        # 기존 전략 인스턴스에 파라미터 적용
        if hasattr(strategy_instance, 'params'):
            # 딕셔너리 합치기 (기존 파라미터는 유지하고 최적 파라미터만 업데이트)
            strategy_instance.params.update(best_params)
            logger.info(f"기존 전략에 최적 파라미터 적용됨")
            return strategy_instance
        else:
            logger.warning("전략 인스턴스에 params 속성이 없습니다")
            return False

# 사용 예시
if __name__ == "__main__":
    from strategy.technical import TechnicalStrategy
    
    # 최적화 실행
    optimizer = HyperparameterOptimizer(TechnicalStrategy, ticker="KRW-BTC", optimization_method='genetic')
    
    # 데이터 준비
    optimizer.prepare_data()
    
    # 최적화 실행
    best_params = optimizer.run_optimization(n_trials=50)
    
    if best_params:
        # 최적 파라미터로 전략 생성
        optimized_strategy = optimizer.apply_best_params()
        print(f"최적화된 전략 생성 완료: {optimized_strategy.get_name()}")
        print(f"최적 파라미터: {optimized_strategy.params}")
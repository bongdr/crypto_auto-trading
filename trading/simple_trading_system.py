# 단순 안정 거래 시스템
import pyupbit
import time
import pandas as pd
from datetime import datetime

class SimpleTradingSystem:
    def __init__(self, balance):
        self.balance = balance
        self.holdings = {}
        self.running = False
        print(f"단순 시스템 초기화: {balance:,}원")
    
    def start_trading(self):
        self.running = True
        print("✅ 거래 시작됨 (BTC 전용)")
        return True
    
    def stop_trading(self):
        self.running = False
        print("✅ 거래 중지됨")
        return True

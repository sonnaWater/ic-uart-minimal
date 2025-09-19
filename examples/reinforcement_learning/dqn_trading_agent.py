"""
🤖 簡單強化學習交易智能體範例
Deep Q-Network for Stock Trading

這個範例展示如何使用DQN算法進行股票交易決策
作者: AI量化交易專家學習平台
日期: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

class TradingEnvironment:
    """
    🏗️ 股票交易環境類
    模擬真實的股票交易環境，提供狀態、動作和獎勵
    """
    
    def __init__(self, data, initial_balance=10000):
        """
        初始化交易環境
        
        Args:
            data (pd.DataFrame): 股票價格數據 (OHLCV)
            initial_balance (float): 初始資金
        """
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """重置環境到初始狀態"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        """
        獲取當前狀態
        
        Returns:
            np.array: 包含價格、技術指標、持倉信息的狀態向量
        """
        if self.current_step >= len(self.data) - 1:
            return np.zeros(10)  # 返回零狀態表示結束
            
        current_price = self.data.iloc[self.current_step]['close']
        
        # 🔍 計算技術指標
        window = min(20, self.current_step + 1)
        prices = self.data.iloc[max(0, self.current_step-window+1):self.current_step+1]['close']
        
        sma_20 = prices.mean()
        price_to_sma = current_price / sma_20 if sma_20 > 0 else 1.0
        
        volatility = prices.pct_change().std() if len(prices) > 1 else 0.0
        
        # 📊 狀態向量: [價格變化率, SMA比率, 波動率, 持倉比例, 現金比例, ...]
        state = np.array([
            self.data.iloc[self.current_step]['close'] / self.data.iloc[max(0, self.current_step-1)]['close'] - 1.0,  # 價格變化率
            price_to_sma - 1.0,  # 價格相對SMA
            volatility,  # 波動率
            self.shares_held * current_price / self.total_value if self.total_value > 0 else 0,  # 持倉比例
            self.balance / self.total_value if self.total_value > 0 else 1,  # 現金比例
            (current_price - self.data.iloc[max(0, self.current_step-5)]['close']) / current_price if current_price > 0 else 0,  # 5日收益率
            self.data.iloc[self.current_step]['volume'] / self.data['volume'].mean() - 1.0,  # 成交量異常
            0.0,  # 預留位置
            0.0,  # 預留位置
            0.0   # 預留位置
        ])
        
        return state
    
    def step(self, action):
        """
        執行動作並返回下一狀態
        
        Args:
            action (int): 0=持有, 1=買入, 2=賣出
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        current_price = self.data.iloc[self.current_step]['close']
        
        # 💰 執行交易動作
        if action == 1:  # 買入
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))
                
        elif action == 2:  # 賣出
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.trades.append(('SELL', self.current_step, current_price, self.shares_held))
                self.shares_held = 0
        
        # 📈 計算獎勵
        previous_value = self.total_value
        self.total_value = self.balance + self.shares_held * current_price
        
        # 🎯 獎勵函數設計
        reward = (self.total_value - previous_value) / previous_value if previous_value > 0 else 0
        
        # 添加風險調整獎勵
        if len(self.trades) >= 2:
            # 計算最近交易的夏普比率獎勵
            returns = [(trade[2] - self.trades[i-1][2]) / self.trades[i-1][2] 
                      for i, trade in enumerate(self.trades[1:], 1)]
            if len(returns) > 1:
                sharpe_bonus = np.mean(returns) / (np.std(returns) + 1e-6) * 0.01
                reward += sharpe_bonus
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        next_state = self._get_state()
        
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_value': self.total_value,
            'current_price': current_price
        }
        
        return next_state, reward, done, info


class DQN(nn.Module):
    """
    🧠 深度Q網路架構
    使用全連接層建構DQN模型
    """
    
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        初始化DQN網路
        
        Args:
            state_size (int): 狀態向量維度
            action_size (int): 動作空間大小
            hidden_size (int): 隱藏層神經元數量
        """
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x):
        """前向傳播"""
        return self.network(x)


class DQNAgent:
    """
    🎯 DQN智能體
    實現經驗回放、目標網路等DQN核心技術
    """
    
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        初始化DQN智能體
        
        Args:
            state_size (int): 狀態空間維度
            action_size (int): 動作空間大小
            lr (float): 學習率
            gamma (float): 折扣因子
            epsilon (float): 探索率
            epsilon_decay (float): 探索率衰減
            epsilon_min (float): 最小探索率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # 經驗回放緩衝區
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 🧠 建立主網路和目標網路
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 初始化目標網路
        self.update_target_network()
        
    def remember(self, state, action, reward, next_state, done):
        """存儲經驗到回放緩衝區"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        根據ε-貪婪策略選擇動作
        
        Args:
            state (np.array): 當前狀態
            
        Returns:
            int: 選擇的動作
        """
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size=32):
        """
        經驗回放訓練
        
        Args:
            batch_size (int): 批次大小
        """
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """更新目標網路權重"""
        self.target_network.load_state_dict(self.q_network.state_dict())


def generate_sample_data(days=1000):
    """
    🎲 生成模擬股票數據
    
    Args:
        days (int): 數據天數
        
    Returns:
        pd.DataFrame: 包含OHLCV的股票數據
    """
    np.random.seed(42)
    
    # 生成價格隨機遊走
    price_changes = np.random.normal(0.001, 0.02, days)  # 日均漲幅0.1%，波動率2%
    prices = [100]  # 初始價格
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # 確保價格為正
    
    # 生成OHLCV數據
    data = []
    for i in range(days):
        close = prices[i+1]
        open_price = prices[i] * np.random.uniform(0.99, 1.01)
        high = max(open_price, close) * np.random.uniform(1.0, 1.02)
        low = min(open_price, close) * np.random.uniform(0.98, 1.0)
        volume = np.random.uniform(1000000, 5000000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def train_dqn_agent(episodes=1000):
    """
    🚀 訓練DQN智能體
    
    Args:
        episodes (int): 訓練回合數
        
    Returns:
        tuple: (訓練好的智能體, 訓練歷史)
    """
    # 生成訓練數據
    print("📊 生成模擬股票數據...")
    data = generate_sample_data(1000)
    
    # 創建環境和智能體
    env = TradingEnvironment(data)
    agent = DQNAgent(state_size=10, action_size=3)
    
    scores = []
    best_score = -np.inf
    
    print(f"🎯 開始訓練 {episodes} 回合...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 訓練智能體
        if len(agent.memory) > 32:
            agent.replay()
        
        # 更新目標網路
        if episode % 100 == 0:
            agent.update_target_network()
        
        scores.append(total_reward)
        
        # 記錄最佳表現
        if total_reward > best_score:
            best_score = total_reward
        
        # 打印進度
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"📈 Episode {episode}: 平均獎勵 = {avg_score:.4f}, 最佳獎勵 = {best_score:.4f}, ε = {agent.epsilon:.3f}")
    
    return agent, scores


def evaluate_agent(agent, test_data):
    """
    📊 評估智能體表現
    
    Args:
        agent: 訓練好的DQN智能體
        test_data: 測試數據
        
    Returns:
        dict: 評估結果
    """
    env = TradingEnvironment(test_data)
    agent.epsilon = 0  # 關閉探索
    
    state = env.reset()
    total_reward = 0
    actions_taken = []
    
    while True:
        action = agent.act(state)
        actions_taken.append(action)
        next_state, reward, done, info = env.step(action)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    final_value = info['total_value']
    roi = (final_value - env.initial_balance) / env.initial_balance
    
    # 計算更多指標
    buy_hold_return = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close']
    
    results = {
        'total_reward': total_reward,
        'final_value': final_value,
        'roi': roi,
        'buy_hold_return': buy_hold_return,
        'excess_return': roi - buy_hold_return,
        'trades': env.trades,
        'actions': actions_taken
    }
    
    return results


if __name__ == "__main__":
    print("🤖 AI量化交易DQN智能體訓練示例")
    print("=" * 50)
    
    # 🚀 訓練智能體
    agent, training_scores = train_dqn_agent(episodes=500)
    
    # 📊 生成測試數據並評估
    print("\n📊 生成測試數據並評估智能體表現...")
    test_data = generate_sample_data(200)
    results = evaluate_agent(agent, test_data)
    
    # 📈 顯示結果
    print("\n🎯 評估結果:")
    print(f"總獎勵: {results['total_reward']:.4f}")
    print(f"投資回報率: {results['roi']:.2%}")
    print(f"買入持有回報率: {results['buy_hold_return']:.2%}")
    print(f"超額收益: {results['excess_return']:.2%}")
    print(f"交易次數: {len(results['trades'])}")
    print(f"最終資產價值: ${results['final_value']:.2f}")
    
    print("\n🏆 恭喜！你已經成功實現了第一個AI交易智能體！")
    print("💪 繼續精進，向專業量化交易專家邁進！")
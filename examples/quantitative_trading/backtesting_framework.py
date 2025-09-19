"""
📊 量化交易策略回測框架
Quantitative Trading Backtesting Framework

這個範例展示如何建立專業的量化交易回測系統
包含多種經典策略和風險管理模組

作者: AI量化交易專家學習平台
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class Portfolio:
    """
    💼 投資組合管理類
    管理資金、持倉、交易記錄等
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        初始化投資組合
        
        Args:
            initial_capital (float): 初始資金
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: shares}
        self.portfolio_value = initial_capital
        self.trades = []
        self.daily_values = []
        self.daily_returns = []
        
    def buy(self, symbol: str, price: float, shares: int, timestamp: datetime) -> bool:
        """
        買入股票
        
        Args:
            symbol (str): 股票代碼
            price (float): 買入價格
            shares (int): 買入股數
            timestamp (datetime): 交易時間
            
        Returns:
            bool: 是否成功執行交易
        """
        cost = price * shares
        if cost <= self.cash:
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + shares
            
            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'BUY',
                'price': price,
                'shares': shares,
                'value': cost
            })
            return True
        return False
    
    def sell(self, symbol: str, price: float, shares: int, timestamp: datetime) -> bool:
        """
        賣出股票
        
        Args:
            symbol (str): 股票代碼  
            price (float): 賣出價格
            shares (int): 賣出股數
            timestamp (datetime): 交易時間
            
        Returns:
            bool: 是否成功執行交易
        """
        if symbol in self.positions and self.positions[symbol] >= shares:
            revenue = price * shares
            self.cash += revenue
            self.positions[symbol] -= shares
            
            if self.positions[symbol] == 0:
                del self.positions[symbol]
            
            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'SELL',
                'price': price,
                'shares': shares,
                'value': revenue
            })
            return True
        return False
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """
        更新投資組合價值
        
        Args:
            current_prices (Dict[str, float]): 當前股票價格
        """
        holdings_value = sum(
            shares * current_prices.get(symbol, 0) 
            for symbol, shares in self.positions.items()
        )
        
        self.portfolio_value = self.cash + holdings_value
        self.daily_values.append(self.portfolio_value)
        
        # 計算日收益率
        if len(self.daily_values) > 1:
            daily_return = (self.portfolio_value - self.daily_values[-2]) / self.daily_values[-2]
            self.daily_returns.append(daily_return)
    
    def get_performance_metrics(self) -> Dict:
        """
        計算投資組合績效指標
        
        Returns:
            Dict: 包含各種績效指標的字典
        """
        if len(self.daily_returns) == 0:
            return {}
        
        returns = np.array(self.daily_returns)
        
        # 🎯 基本收益指標
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 📊 風險指標
        volatility = np.std(returns) * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 🏆 風險調整收益指標
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        # 📉 回撤分析
        cumulative_returns = (1 + pd.Series(returns)).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 🎪 其他指標
        win_rate = len(returns[returns > 0]) / len(returns)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_value': self.portfolio_value
        }


class MovingAverageStrategy:
    """
    📈 移動平均線交叉策略
    經典的趨勢跟隨策略
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        初始化策略參數
        
        Args:
            short_window (int): 短期移動平均窗口
            long_window (int): 長期移動平均窗口
        """
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"MA_{short_window}_{long_window}"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信號
        
        Args:
            data (pd.DataFrame): 股票價格數據
            
        Returns:
            pd.DataFrame: 包含交易信號的數據
        """
        signals = data.copy()
        
        # 計算移動平均線
        signals[f'MA_{self.short_window}'] = signals['close'].rolling(window=self.short_window).mean()
        signals[f'MA_{self.long_window}'] = signals['close'].rolling(window=self.long_window).mean()
        
        # 生成交易信號
        signals['signal'] = 0
        signals['signal'][self.short_window:] = np.where(
            signals[f'MA_{self.short_window}'][self.short_window:] > 
            signals[f'MA_{self.long_window}'][self.short_window:], 1, 0
        )
        
        # 生成交易位置
        signals['position'] = signals['signal'].diff()
        
        return signals


class MeanReversionStrategy:
    """
    🔄 均值回歸策略
    基於布林帶的反轉策略
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        """
        初始化策略參數
        
        Args:
            window (int): 移動平均窗口
            num_std (float): 標準差倍數
        """
        self.window = window
        self.num_std = num_std
        self.name = f"MeanReversion_{window}_{num_std}"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信號
        
        Args:
            data (pd.DataFrame): 股票價格數據
            
        Returns:
            pd.DataFrame: 包含交易信號的數據
        """
        signals = data.copy()
        
        # 計算布林帶
        signals['MA'] = signals['close'].rolling(window=self.window).mean()
        signals['std'] = signals['close'].rolling(window=self.window).std()
        signals['upper_band'] = signals['MA'] + (signals['std'] * self.num_std)
        signals['lower_band'] = signals['MA'] - (signals['std'] * self.num_std)
        
        # 生成交易信號
        signals['signal'] = 0
        
        # 當價格觸及下軌時買入，觸及上軌時賣出
        buy_signals = signals['close'] <= signals['lower_band']
        sell_signals = signals['close'] >= signals['upper_band']
        
        signals.loc[buy_signals, 'signal'] = 1  # 買入信號
        signals.loc[sell_signals, 'signal'] = -1  # 賣出信號
        
        # 生成交易位置
        signals['position'] = signals['signal']
        
        return signals


class MomentumStrategy:
    """
    🚀 動量策略
    基於價格動量的趨勢跟隨策略
    """
    
    def __init__(self, lookback_period: int = 12, holding_period: int = 1):
        """
        初始化策略參數
        
        Args:
            lookback_period (int): 動量計算回看期
            holding_period (int): 持有期
        """
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.name = f"Momentum_{lookback_period}_{holding_period}"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信號
        
        Args:
            data (pd.DataFrame): 股票價格數據
            
        Returns:
            pd.DataFrame: 包含交易信號的數據
        """
        signals = data.copy()
        
        # 計算動量指標
        signals['momentum'] = signals['close'].pct_change(periods=self.lookback_period)
        
        # 生成交易信號 (動量>0買入，動量<0賣出)
        signals['signal'] = 0
        signals['signal'] = np.where(signals['momentum'] > 0, 1, 0)
        signals['signal'] = np.where(signals['momentum'] < 0, -1, signals['signal'])
        
        # 生成交易位置
        signals['position'] = signals['signal']
        
        return signals


class Backtester:
    """
    🔍 回測引擎
    執行策略回測並生成績效報告
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        初始化回測引擎
        
        Args:
            initial_capital (float): 初始資金
            commission (float): 交易手續費率
        """
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run_backtest(self, strategy, data: pd.DataFrame, symbol: str = "STOCK") -> Dict:
        """
        運行策略回測
        
        Args:
            strategy: 交易策略對象
            data (pd.DataFrame): 股票價格數據
            symbol (str): 股票代碼
            
        Returns:
            Dict: 回測結果
        """
        # 生成交易信號
        signals = strategy.generate_signals(data)
        
        # 初始化投資組合
        portfolio = Portfolio(self.initial_capital)
        
        # 執行回測
        position = 0  # 當前持倉狀態
        
        for i, row in signals.iterrows():
            current_price = row['close']
            current_time = row.name if hasattr(row.name, 'date') else datetime.now()
            
            # 更新投資組合價值
            current_prices = {symbol: current_price}
            portfolio.update_portfolio_value(current_prices)
            
            # 檢查交易信號
            if hasattr(row, 'position') and pd.notna(row['position']):
                if row['position'] == 1 and position <= 0:  # 買入信號
                    # 計算可買入股數
                    available_cash = portfolio.cash * 0.95  # 保留5%現金
                    shares_to_buy = int(available_cash / (current_price * (1 + self.commission)))
                    
                    if shares_to_buy > 0:
                        actual_cost = shares_to_buy * current_price * (1 + self.commission)
                        if portfolio.buy(symbol, current_price, shares_to_buy, current_time):
                            portfolio.cash -= actual_cost - (shares_to_buy * current_price)  # 扣除手續費
                            position = 1
                
                elif row['position'] == -1 and position >= 0:  # 賣出信號
                    if symbol in portfolio.positions and portfolio.positions[symbol] > 0:
                        shares_to_sell = portfolio.positions[symbol]
                        if portfolio.sell(symbol, current_price, shares_to_sell, current_time):
                            # 扣除手續費
                            commission_cost = shares_to_sell * current_price * self.commission
                            portfolio.cash -= commission_cost
                            position = -1
        
        # 計算績效指標
        performance = portfolio.get_performance_metrics()
        
        # 計算基準收益 (買入持有)
        buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        
        # 添加額外信息
        performance['strategy_name'] = strategy.name
        performance['buy_hold_return'] = buy_hold_return
        performance['excess_return'] = performance['total_return'] - buy_hold_return
        performance['portfolio'] = portfolio
        performance['signals'] = signals
        
        return performance


def generate_sample_stock_data(days: int = 1000, initial_price: float = 100) -> pd.DataFrame:
    """
    🎲 生成模擬股票數據
    
    Args:
        days (int): 數據天數
        initial_price (float): 初始價格
        
    Returns:
        pd.DataFrame: 股票價格數據
    """
    np.random.seed(42)
    
    # 生成日期索引
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    # 生成價格隨機遊走 (幾何布朗運動)
    mu = 0.0005  # 日均收益率
    sigma = 0.02  # 日波動率
    
    price_changes = np.random.normal(mu, sigma, days)
    prices = [initial_price]
    
    for change in price_changes:
        new_price = prices[-1] * np.exp(change)
        prices.append(new_price)
    
    # 生成OHLCV數據
    data = []
    for i in range(days):
        close = prices[i+1]
        open_price = prices[i] * np.random.uniform(0.995, 1.005)
        
        daily_volatility = abs(np.random.normal(0, sigma))
        high = max(open_price, close) * (1 + daily_volatility)
        low = min(open_price, close) * (1 - daily_volatility)
        
        volume = np.random.uniform(1000000, 5000000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def plot_backtest_results(results: Dict):
    """
    📊 繪製回測結果圖表
    
    Args:
        results (Dict): 回測結果
    """
    portfolio = results['portfolio']
    signals = results['signals']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'{results["strategy_name"]} 策略回測結果', fontsize=16, fontweight='bold')
    
    # 1. 股價與交易信號
    ax1 = axes[0, 0]
    ax1.plot(signals.index, signals['close'], label='股價', alpha=0.7)
    
    if 'MA_20' in signals.columns:
        ax1.plot(signals.index, signals['MA_20'], label='MA20', alpha=0.7)
    if 'MA_50' in signals.columns:
        ax1.plot(signals.index, signals['MA_50'], label='MA50', alpha=0.7)
    
    # 標記買賣點
    buy_signals = signals[signals['position'] == 1]
    sell_signals = signals[signals['position'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], 
               color='green', marker='^', s=60, label='買入', alpha=0.8)
    ax1.scatter(sell_signals.index, sell_signals['close'], 
               color='red', marker='v', s=60, label='賣出', alpha=0.8)
    
    ax1.set_title('股價與交易信號')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 投資組合價值變化
    ax2 = axes[0, 1]
    portfolio_values = portfolio.daily_values
    dates = signals.index[:len(portfolio_values)]
    
    ax2.plot(dates, portfolio_values, label='投資組合價值', color='blue')
    ax2.axhline(y=portfolio.initial_capital, color='red', linestyle='--', alpha=0.7, label='初始資金')
    ax2.set_title('投資組合價值變化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 累積收益率對比
    ax3 = axes[1, 0]
    
    portfolio_returns = np.array(portfolio.daily_returns)
    strategy_cumulative = (1 + pd.Series(portfolio_returns)).cumprod()
    
    stock_returns = signals['close'].pct_change().dropna()
    buy_hold_cumulative = (1 + stock_returns).cumprod()
    
    dates_returns = signals.index[1:len(strategy_cumulative)+1]
    
    ax3.plot(dates_returns, strategy_cumulative, label=f'{results["strategy_name"]}策略', color='blue')
    ax3.plot(dates_returns, buy_hold_cumulative, label='買入持有', color='orange')
    ax3.set_title('累積收益率對比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 回撤分析
    ax4 = axes[1, 1]
    
    rolling_max = pd.Series(strategy_cumulative).expanding().max()
    drawdowns = (strategy_cumulative - rolling_max) / rolling_max
    
    ax4.fill_between(dates_returns, drawdowns, 0, color='red', alpha=0.3)
    ax4.plot(dates_returns, drawdowns, color='red')
    ax4.set_title(f'策略回撤分析 (最大回撤: {results["max_drawdown"]:.2%})')
    ax4.grid(True, alpha=0.3)
    
    # 5. 績效指標表
    ax5 = axes[2, 0]
    ax5.axis('off')
    
    metrics_text = f"""
    📊 績效指標總覽:
    
    🎯 總收益率: {results['total_return']:.2%}
    📈 年化收益率: {results['annual_return']:.2%}
    📊 波動率: {results['volatility']:.2%}
    🏆 夏普比率: {results['sharpe_ratio']:.2f}
    🎪 索提諾比率: {results['sortino_ratio']:.2f}
    📉 最大回撤: {results['max_drawdown']:.2%}
    🎯 勝率: {results['win_rate']:.2%}
    💰 交易次數: {results['total_trades']}
    🚀 超額收益: {results['excess_return']:.2%}
    💎 最終價值: ${results['final_value']:,.2f}
    """
    
    ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    # 6. 交易分布
    ax6 = axes[2, 1]
    
    if portfolio.trades:
        trades_df = pd.DataFrame(portfolio.trades)
        trade_pnl = []
        
        for i in range(0, len(trades_df), 2):
            if i + 1 < len(trades_df):
                buy_trade = trades_df.iloc[i]
                sell_trade = trades_df.iloc[i + 1]
                if buy_trade['action'] == 'BUY' and sell_trade['action'] == 'SELL':
                    pnl = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                    trade_pnl.append(pnl)
        
        if trade_pnl:
            ax6.hist(trade_pnl, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax6.set_title('單筆交易收益分布')
            ax6.set_xlabel('收益率')
            ax6.set_ylabel('頻次')
            ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_strategy_comparison():
    """
    🏆 運行多策略對比分析
    """
    print("📊 生成模擬股票數據...")
    data = generate_sample_stock_data(days=1000)
    
    # 初始化策略
    strategies = [
        MovingAverageStrategy(short_window=20, long_window=50),
        MeanReversionStrategy(window=20, num_std=2.0),
        MomentumStrategy(lookback_period=20, holding_period=5)
    ]
    
    # 初始化回測引擎
    backtester = Backtester(initial_capital=100000, commission=0.001)
    
    results = {}
    
    print("🚀 開始策略回測...")
    for strategy in strategies:
        print(f"正在回測 {strategy.name} 策略...")
        result = backtester.run_backtest(strategy, data)
        results[strategy.name] = result
    
    # 生成對比報告
    print("\n" + "="*80)
    print("📊 策略績效對比報告")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        name: {
            '總收益率': f"{result['total_return']:.2%}",
            '年化收益率': f"{result['annual_return']:.2%}",
            '夏普比率': f"{result['sharpe_ratio']:.2f}",
            '最大回撤': f"{result['max_drawdown']:.2%}",
            '勝率': f"{result['win_rate']:.2%}",
            '交易次數': result['total_trades'],
            '超額收益': f"{result['excess_return']:.2%}"
        }
        for name, result in results.items()
    })
    
    print(comparison_df.to_string())
    
    # 繪製對比圖表
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # 收益率對比
    strategies_names = list(results.keys())
    total_returns = [results[name]['total_return'] for name in strategies_names]
    sharpe_ratios = [results[name]['sharpe_ratio'] for name in strategies_names]
    
    ax[0].bar(strategies_names, total_returns, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax[0].set_title('策略總收益率對比')
    ax[0].set_ylabel('總收益率')
    ax[0].tick_params(axis='x', rotation=45)
    
    # 夏普比率對比
    ax[1].bar(strategies_names, sharpe_ratios, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax[1].set_title('策略夏普比率對比')
    ax[1].set_ylabel('夏普比率')
    ax[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 為最佳策略生成詳細圖表
    best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
    print(f"\n🏆 最佳策略: {best_strategy} (夏普比率: {results[best_strategy]['sharpe_ratio']:.2f})")
    
    plot_backtest_results(results[best_strategy])
    
    return results


if __name__ == "__main__":
    print("💹 量化交易策略回測框架")
    print("🎯 專業回測系統 - 從零到專家")
    print("="*60)
    
    # 運行策略對比
    results = run_strategy_comparison()
    
    print("\n🎊 恭喜！你已經建立了專業的量化交易回測系統！")
    print("💪 這是成為量化交易專家的重要一步！")
    print("🚀 繼續精進，探索更多高級策略！")
"""
📊 金融數據分析工具集
Financial Data Analysis Toolkit

這個範例展示如何進行專業的金融數據分析
包含技術指標計算、統計分析、可視化等功能

作者: AI量化交易專家學習平台
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class TechnicalIndicators:
    """
    📈 技術指標計算類
    實現各種常用的技術分析指標
    """
    
    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """簡單移動平均"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def ema(prices: pd.Series, window: int) -> pd.Series:
        """指數移動平均"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """布林帶"""
        sma = TechnicalIndicators.sma(prices, window)
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """相對強弱指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD指標"""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> tuple:
        """隨機震盪指標"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """平均真實範圍"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr


class StatisticalAnalysis:
    """
    📊 統計分析類
    提供各種統計分析功能
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """計算收益率"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """計算對數收益率"""
        return np.log(prices / prices.shift(1)).dropna()
    
    @staticmethod
    def rolling_statistics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """滾動統計指標"""
        stats_df = pd.DataFrame(index=returns.index)
        
        # 滾動年化收益率
        stats_df['rolling_return'] = returns.rolling(window=window).mean() * 252
        
        # 滾動波動率
        stats_df['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # 滾動夏普比率
        stats_df['rolling_sharpe'] = stats_df['rolling_return'] / stats_df['rolling_volatility']
        
        # 滾動最大回撤
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(window=window, min_periods=1).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        stats_df['rolling_max_drawdown'] = drawdown.rolling(window=window).min()
        
        return stats_df
    
    @staticmethod
    def performance_metrics(returns: pd.Series) -> dict:
        """績效指標計算"""
        metrics = {}
        
        # 基本統計
        metrics['mean_return'] = returns.mean()
        metrics['std_return'] = returns.std()
        metrics['skewness'] = stats.skew(returns.dropna())
        metrics['kurtosis'] = stats.kurtosis(returns.dropna())
        
        # 年化指標
        metrics['annual_return'] = returns.mean() * 252
        metrics['annual_volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_volatility'] if metrics['annual_volatility'] > 0 else 0
        
        # 風險指標
        negative_returns = returns[returns < 0]
        metrics['downside_deviation'] = negative_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = metrics['annual_return'] / metrics['downside_deviation'] if metrics['downside_deviation'] > 0 else 0
        
        # VaR和CVaR
        metrics['var_95'] = np.percentile(returns.dropna(), 5)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        return metrics
    
    @staticmethod
    def correlation_analysis(returns_df: pd.DataFrame) -> pd.DataFrame:
        """相關性分析"""
        correlation_matrix = returns_df.corr()
        return correlation_matrix
    
    @staticmethod
    def normality_test(returns: pd.Series) -> dict:
        """正態性檢驗"""
        results = {}
        
        # Shapiro-Wilk檢驗
        shapiro_stat, shapiro_p = stats.shapiro(returns.dropna())
        results['shapiro_stat'] = shapiro_stat
        results['shapiro_p_value'] = shapiro_p
        
        # Jarque-Bera檢驗
        jb_stat, jb_p = stats.jarque_bera(returns.dropna())
        results['jarque_bera_stat'] = jb_stat
        results['jarque_bera_p_value'] = jb_p
        
        # Kolmogorov-Smirnov檢驗
        ks_stat, ks_p = stats.kstest(returns.dropna(), 'norm', 
                                   args=(returns.mean(), returns.std()))
        results['ks_stat'] = ks_stat
        results['ks_p_value'] = ks_p
        
        return results


class DataVisualizer:
    """
    📊 數據可視化類
    提供各種專業的金融數據可視化功能
    """
    
    @staticmethod
    def plot_price_and_volume(data: pd.DataFrame, title: str = "股價與成交量"):
        """繪製價格和成交量圖"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # 價格圖
        ax1.plot(data.index, data['close'], label='收盤價', linewidth=1.5)
        if 'open' in data.columns:
            ax1.plot(data.index, data['open'], label='開盤價', alpha=0.7)
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('價格')
        
        # 成交量圖
        if 'volume' in data.columns:
            ax2.bar(data.index, data['volume'], alpha=0.7, color='orange')
            ax2.set_title('成交量')
            ax2.set_ylabel('成交量')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_technical_indicators(data: pd.DataFrame, indicators: dict):
        """繪製技術指標圖"""
        n_indicators = len(indicators)
        fig, axes = plt.subplots(n_indicators + 1, 1, figsize=(12, (n_indicators + 1) * 3))
        
        if n_indicators == 0:
            axes = [axes]
        
        # 主圖：價格
        axes[0].plot(data.index, data['close'], label='收盤價', linewidth=1.5)
        
        for name, values in indicators.items():
            if name.startswith('BB'):  # 布林帶
                upper, middle, lower = values
                axes[0].plot(data.index, upper, label='布林帶上軌', alpha=0.7)
                axes[0].plot(data.index, middle, label='布林帶中軌', alpha=0.7)
                axes[0].plot(data.index, lower, label='布林帶下軌', alpha=0.7)
                axes[0].fill_between(data.index, upper, lower, alpha=0.1)
            elif name in ['SMA', 'EMA']:
                for period, line in values.items():
                    axes[0].plot(data.index, line, label=f'{name}{period}', alpha=0.8)
        
        axes[0].set_title('價格與趨勢指標')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 子圖：震盪指標
        subplot_idx = 1
        for name, values in indicators.items():
            if name == 'RSI':
                axes[subplot_idx].plot(data.index, values, label='RSI', color='purple')
                axes[subplot_idx].axhline(y=70, color='r', linestyle='--', alpha=0.7)
                axes[subplot_idx].axhline(y=30, color='g', linestyle='--', alpha=0.7)
                axes[subplot_idx].set_title('RSI指標')
                axes[subplot_idx].set_ylabel('RSI')
                subplot_idx += 1
            
            elif name == 'MACD':
                macd_line, signal_line, histogram = values
                axes[subplot_idx].plot(data.index, macd_line, label='MACD', color='blue')
                axes[subplot_idx].plot(data.index, signal_line, label='Signal', color='red')
                axes[subplot_idx].bar(data.index, histogram, label='Histogram', alpha=0.6, color='grey')
                axes[subplot_idx].set_title('MACD指標')
                axes[subplot_idx].legend()
                subplot_idx += 1
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_returns_analysis(returns: pd.Series, title: str = "收益率分析"):
        """繪製收益率分析圖"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 收益率時間序列
        axes[0, 0].plot(returns.index, returns, alpha=0.7)
        axes[0, 0].set_title('收益率時間序列')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylabel('收益率')
        
        # 收益率分布直方圖
        axes[0, 1].hist(returns.dropna(), bins=50, alpha=0.7, density=True, color='skyblue')
        
        # 擬合正態分布
        mu, sigma = stats.norm.fit(returns.dropna())
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, mu, sigma)
        axes[0, 1].plot(x, y, 'r-', linewidth=2, label=f'正態分布 (μ={mu:.4f}, σ={sigma:.4f})')
        
        axes[0, 1].set_title('收益率分布')
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('收益率')
        axes[0, 1].set_ylabel('密度')
        
        # Q-Q圖
        stats.probplot(returns.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q圖 (正態性檢驗)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 累積收益率
        cumulative_returns = (1 + returns).cumprod()
        axes[1, 1].plot(cumulative_returns.index, cumulative_returns)
        axes[1, 1].set_title('累積收益率')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylabel('累積收益')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str = "相關性熱力圖"):
        """繪製相關性熱力圖"""
        plt.figure(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(correlation_matrix))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_rolling_statistics(stats_df: pd.DataFrame, title: str = "滾動統計指標"):
        """繪製滾動統計指標"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 滾動收益率
        axes[0, 0].plot(stats_df.index, stats_df['rolling_return'])
        axes[0, 0].set_title('滾動年化收益率')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylabel('年化收益率')
        
        # 滾動波動率
        axes[0, 1].plot(stats_df.index, stats_df['rolling_volatility'], color='orange')
        axes[0, 1].set_title('滾動年化波動率')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylabel('年化波動率')
        
        # 滾動夏普比率
        axes[1, 0].plot(stats_df.index, stats_df['rolling_sharpe'], color='green')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('滾動夏普比率')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylabel('夏普比率')
        
        # 滾動最大回撤
        axes[1, 1].fill_between(stats_df.index, stats_df['rolling_max_drawdown'], 0, 
                               alpha=0.7, color='red')
        axes[1, 1].set_title('滾動最大回撤')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylabel('最大回撤')
        
        plt.tight_layout()
        plt.show()


def generate_sample_multi_asset_data(assets: list, days: int = 1000) -> pd.DataFrame:
    """
    🎲 生成多資產模擬數據
    
    Args:
        assets (list): 資產名稱列表
        days (int): 數據天數
        
    Returns:
        pd.DataFrame: 多資產價格數據
    """
    np.random.seed(42)
    
    # 生成日期索引
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    # 設置不同資產的參數
    asset_params = {
        'Stock_A': {'mu': 0.0008, 'sigma': 0.02, 'initial_price': 100},
        'Stock_B': {'mu': 0.0006, 'sigma': 0.025, 'initial_price': 80},
        'Stock_C': {'mu': 0.0004, 'sigma': 0.018, 'initial_price': 120},
        'Bond': {'mu': 0.0002, 'sigma': 0.005, 'initial_price': 100},
        'Commodity': {'mu': 0.0005, 'sigma': 0.03, 'initial_price': 50}
    }
    
    data = pd.DataFrame(index=dates)
    
    for asset in assets:
        if asset in asset_params:
            params = asset_params[asset]
        else:
            params = {'mu': 0.0005, 'sigma': 0.02, 'initial_price': 100}
        
        # 生成價格隨機遊走
        price_changes = np.random.normal(params['mu'], params['sigma'], days)
        prices = [params['initial_price']]
        
        for change in price_changes:
            new_price = prices[-1] * np.exp(change)
            prices.append(new_price)
        
        data[asset] = prices[1:]
    
    return data


def comprehensive_analysis_example():
    """
    🔍 綜合分析範例
    展示完整的金融數據分析流程
    """
    print("📊 金融數據分析工具集演示")
    print("=" * 50)
    
    # 1. 生成模擬數據
    print("🎲 生成模擬多資產數據...")
    assets = ['Stock_A', 'Stock_B', 'Stock_C', 'Bond', 'Commodity']
    price_data = generate_sample_multi_asset_data(assets, days=1000)
    
    # 2. 技術指標分析
    print("📈 計算技術指標...")
    stock_a_data = pd.DataFrame({
        'close': price_data['Stock_A'],
        'high': price_data['Stock_A'] * 1.02,
        'low': price_data['Stock_A'] * 0.98,
        'volume': np.random.uniform(1000000, 5000000, len(price_data))
    })
    
    # 計算各種技術指標
    indicators = {}
    indicators['SMA'] = {
        20: TechnicalIndicators.sma(stock_a_data['close'], 20),
        50: TechnicalIndicators.sma(stock_a_data['close'], 50)
    }
    indicators['RSI'] = TechnicalIndicators.rsi(stock_a_data['close'])
    indicators['MACD'] = TechnicalIndicators.macd(stock_a_data['close'])
    indicators['BB'] = TechnicalIndicators.bollinger_bands(stock_a_data['close'])
    
    # 3. 收益率計算
    print("💰 計算收益率...")
    returns_data = pd.DataFrame()
    for asset in assets:
        returns_data[asset] = StatisticalAnalysis.calculate_returns(price_data[asset])
    
    # 4. 統計分析
    print("📊 進行統計分析...")
    performance_metrics = {}
    for asset in assets:
        performance_metrics[asset] = StatisticalAnalysis.performance_metrics(returns_data[asset])
    
    # 5. 相關性分析
    print("🔗 進行相關性分析...")
    correlation_matrix = StatisticalAnalysis.correlation_analysis(returns_data)
    
    # 6. 滾動統計
    print("📈 計算滾動統計指標...")
    rolling_stats = StatisticalAnalysis.rolling_statistics(returns_data['Stock_A'])
    
    # 7. 正態性檢驗
    print("🔍 進行正態性檢驗...")
    normality_results = {}
    for asset in assets:
        normality_results[asset] = StatisticalAnalysis.normality_test(returns_data[asset])
    
    # 8. 結果展示
    print("\n" + "="*80)
    print("📊 分析結果總覽")
    print("="*80)
    
    # 績效指標總覽
    metrics_df = pd.DataFrame(performance_metrics).T
    print("\n🎯 績效指標總覽:")
    print(metrics_df.round(4))
    
    # 相關性矩陣
    print("\n🔗 資產相關性矩陣:")
    print(correlation_matrix.round(3))
    
    # 正態性檢驗結果
    print("\n🔍 正態性檢驗結果 (p值):")
    normality_df = pd.DataFrame({
        asset: {
            'Shapiro-Wilk': results['shapiro_p_value'],
            'Jarque-Bera': results['jarque_bera_p_value'],
            'Kolmogorov-Smirnov': results['ks_p_value']
        }
        for asset, results in normality_results.items()
    })
    print(normality_df.round(4))
    
    # 9. 可視化
    print("\n📊 生成可視化圖表...")
    
    # 價格和成交量圖
    DataVisualizer.plot_price_and_volume(stock_a_data, "Stock A 價格與成交量")
    
    # 技術指標圖
    DataVisualizer.plot_technical_indicators(stock_a_data, indicators)
    
    # 收益率分析圖
    DataVisualizer.plot_returns_analysis(returns_data['Stock_A'], "Stock A 收益率分析")
    
    # 相關性熱力圖
    DataVisualizer.plot_correlation_heatmap(correlation_matrix, "資產相關性熱力圖")
    
    # 滾動統計圖
    DataVisualizer.plot_rolling_statistics(rolling_stats, "Stock A 滾動統計指標")
    
    # 10. 投資組合分析
    print("\n💼 投資組合分析...")
    
    # 等權重投資組合
    portfolio_returns = returns_data.mean(axis=1)
    portfolio_metrics = StatisticalAnalysis.performance_metrics(portfolio_returns)
    
    print("🏆 等權重投資組合績效:")
    for metric, value in portfolio_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 可視化投資組合表現
    DataVisualizer.plot_returns_analysis(portfolio_returns, "等權重投資組合收益率分析")
    
    print("\n🎊 分析完成！")
    print("💪 你已經掌握了專業的金融數據分析技能！")
    print("🚀 繼續探索更多高級分析技術！")
    
    return {
        'price_data': price_data,
        'returns_data': returns_data,
        'performance_metrics': performance_metrics,
        'correlation_matrix': correlation_matrix,
        'portfolio_metrics': portfolio_metrics
    }


if __name__ == "__main__":
    print("📊 金融數據分析工具集")
    print("🎯 從數據到洞察 - 專業分析技能")
    print("="*60)
    
    # 運行綜合分析範例
    results = comprehensive_analysis_example()
    
    print("\n🏆 恭喜！你已經建立了完整的金融數據分析工具集！")
    print("📈 這些技能將為你的量化交易之路奠定堅實基礎！")
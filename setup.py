#!/usr/bin/env python3
"""
🚀 AI深度強化學習與量化交易學習平台 - 快速設置腳本
Quick Setup Script for AI-Powered Quantitative Trading Learning Platform

這個腳本幫助用戶快速設置學習環境和依賴包
作者: AI量化交易專家學習平台
日期: 2024
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """打印歡迎橫幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  🧠 AI深度強化學習與量化交易學習平台                               ║
    ║  AI-Powered Quantitative Trading Learning Platform          ║
    ║                                                              ║
    ║  🎯 目標: 成為世界級的AI量化交易專家                              ║
    ║  💪 精神: 在嚴格標準下不斷精進                                   ║
    ║  🚀 願景: 建立一人公司的技術實力                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """檢查Python版本"""
    print("🐍 檢查Python版本...")
    version = sys.version_info
    
    if version.major != 3 or version.minor < 8:
        print("❌ 錯誤: 需要Python 3.8或更高版本")
        print(f"   當前版本: {version.major}.{version.minor}.{version.micro}")
        print("   請升級Python版本後重新運行")
        sys.exit(1)
    
    print(f"✅ Python版本檢查通過: {version.major}.{version.minor}.{version.micro}")

def check_system_requirements():
    """檢查系統需求"""
    print("\n💻 檢查系統環境...")
    
    system = platform.system()
    print(f"   作業系統: {system} {platform.release()}")
    print(f"   處理器架構: {platform.machine()}")
    
    # 檢查可用記憶體 (簡單估算)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   系統記憶體: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("⚠️  警告: 建議至少8GB記憶體以獲得最佳體驗")
    except ImportError:
        print("   記憶體檢查: 跳過 (需要psutil包)")
    
    print("✅ 系統環境檢查完成")

def create_virtual_environment():
    """創建虛擬環境"""
    print("\n🏗️  創建Python虛擬環境...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("   虛擬環境已存在，跳過創建")
        return
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ 虛擬環境創建成功")
        
        # 根據作業系統提供啟動指令
        system = platform.system()
        if system == "Windows":
            activation_cmd = "venv\\Scripts\\activate"
        else:
            activation_cmd = "source venv/bin/activate"
        
        print(f"💡 啟動虛擬環境: {activation_cmd}")
        
    except subprocess.CalledProcessError:
        print("❌ 虛擬環境創建失敗")
        print("   請手動運行: python -m venv venv")

def install_core_dependencies():
    """安裝核心依賴包"""
    print("\n📦 安裝核心依賴包...")
    
    # 核心依賴包 (最小化安裝)
    core_packages = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "jupyter>=1.0.0"
    ]
    
    print("   正在安裝核心包 (這可能需要幾分鐘)...")
    
    try:
        # 升級pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # 安裝核心包
        for package in core_packages:
            print(f"   安裝 {package.split('>=')[0]}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
        
        print("✅ 核心依賴包安裝完成")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依賴包安裝失敗: {e}")
        print("   請嘗試手動安裝: pip install -r requirements.txt")

def setup_project_structure():
    """設置項目結構"""
    print("\n📁 設置項目結構...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models/saved_models",
        "models/checkpoints",
        "notebooks/tutorials",
        "notebooks/experiments",
        "results/backtests",
        "results/reports",
        "logs",
        "configs",
        "tests"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 創建.gitkeep文件以保持空目錄
        gitkeep_file = dir_path / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
    
    print("✅ 項目結構設置完成")

def create_sample_notebook():
    """創建示例筆記本"""
    print("\n📓 創建示例Jupyter筆記本...")
    
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧠 AI量化交易學習平台 - 入門教程\\n",
    "\\n",
    "歡迎來到AI深度強化學習與量化交易學習平台！\\n",
    "\\n",
    "## 🎯 學習目標\\n",
    "- 掌握基礎的數據處理技能\\n",
    "- 理解金融時間序列分析\\n",
    "- 實現簡單的交易策略\\n",
    "- 建立回測評估系統"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 導入必要的庫\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# 設置圖表樣式\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')\\n",
    "\\n",
    "print('🎉 歡迎開始你的AI量化交易學習之旅！')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📊 第一步：生成模擬股票數據"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 生成模擬股票價格數據\\n",
    "np.random.seed(42)\\n",
    "days = 1000\\n",
    "initial_price = 100\\n",
    "\\n",
    "# 使用幾何布朗運動模擬股價\\n",
    "returns = np.random.normal(0.001, 0.02, days)\\n",
    "prices = [initial_price]\\n",
    "\\n",
    "for ret in returns:\\n",
    "    new_price = prices[-1] * (1 + ret)\\n",
    "    prices.append(new_price)\\n",
    "\\n",
    "# 創建DataFrame\\n",
    "dates = pd.date_range('2021-01-01', periods=days+1, freq='D')\\n",
    "stock_data = pd.DataFrame({\\n",
    "    'price': prices,\\n",
    "    'returns': [0] + list(returns)\\n",
    "}, index=dates)\\n",
    "\\n",
    "print(f'✅ 生成了 {len(stock_data)} 天的股票數據')\\n",
    "stock_data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📈 第二步：數據可視化"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 繪製股價圖\\n",
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))\\n",
    "\\n",
    "# 股價趨勢\\n",
    "ax1.plot(stock_data.index, stock_data['price'], label='股價', linewidth=1.5)\\n",
    "ax1.set_title('股價變化趨勢', fontsize=14, fontweight='bold')\\n",
    "ax1.set_ylabel('價格')\\n",
    "ax1.legend()\\n",
    "ax1.grid(True, alpha=0.3)\\n",
    "\\n",
    "# 收益率分布\\n",
    "ax2.hist(stock_data['returns'].dropna(), bins=50, alpha=0.7, density=True)\\n",
    "ax2.set_title('日收益率分布', fontsize=14, fontweight='bold')\\n",
    "ax2.set_xlabel('收益率')\\n",
    "ax2.set_ylabel('密度')\\n",
    "ax2.grid(True, alpha=0.3)\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()\\n",
    "\\n",
    "print('📊 股票數據可視化完成！')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎯 第三步：實現簡單移動平均策略"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 計算移動平均線\\n",
    "stock_data['MA_20'] = stock_data['price'].rolling(window=20).mean()\\n",
    "stock_data['MA_50'] = stock_data['price'].rolling(window=50).mean()\\n",
    "\\n",
    "# 生成交易信號\\n",
    "stock_data['signal'] = 0\\n",
    "stock_data['signal'][20:] = np.where(\\n",
    "    stock_data['MA_20'][20:] > stock_data['MA_50'][20:], 1, 0\\n",
    ")\\n",
    "\\n",
    "# 計算策略收益\\n",
    "stock_data['strategy_returns'] = stock_data['signal'].shift(1) * stock_data['returns']\\n",
    "stock_data['cumulative_returns'] = (1 + stock_data['returns']).cumprod()\\n",
    "stock_data['cumulative_strategy'] = (1 + stock_data['strategy_returns'].fillna(0)).cumprod()\\n",
    "\\n",
    "print('🚀 移動平均交叉策略實現完成！')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📊 第四步：策略績效評估"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 繪製策略表現\\n",
    "plt.figure(figsize=(12, 6))\\n",
    "\\n",
    "plt.plot(stock_data.index, stock_data['cumulative_returns'], \\n",
    "         label='買入持有策略', linewidth=2)\\n",
    "plt.plot(stock_data.index, stock_data['cumulative_strategy'], \\n",
    "         label='移動平均交叉策略', linewidth=2)\\n",
    "\\n",
    "plt.title('策略績效比較', fontsize=14, fontweight='bold')\\n",
    "plt.xlabel('日期')\\n",
    "plt.ylabel('累積收益')\\n",
    "plt.legend()\\n",
    "plt.grid(True, alpha=0.3)\\n",
    "plt.show()\\n",
    "\\n",
    "# 計算績效指標\\n",
    "total_return_bh = stock_data['cumulative_returns'].iloc[-1] - 1\\n",
    "total_return_strategy = stock_data['cumulative_strategy'].iloc[-1] - 1\\n",
    "\\n",
    "print(f'📈 買入持有策略總收益: {total_return_bh:.2%}')\\n",
    "print(f'🎯 移動平均策略總收益: {total_return_strategy:.2%}')\\n",
    "print(f'💪 超額收益: {total_return_strategy - total_return_bh:.2%}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎊 恭喜！\\n",
    "\\n",
    "你已經完成了第一個量化交易策略的實現！\\n",
    "\\n",
    "### 📚 下一步學習方向：\\n",
    "1. 學習更多技術指標\\n",
    "2. 實現風險管理模組\\n",
    "3. 探索機器學習策略\\n",
    "4. 深入強化學習算法\\n",
    "\\n",
    "### 💪 俄羅斯裁判的鼓勵：\\n",
    "*同志，這只是開始！繼續努力，追求卓越！*"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    notebook_path = Path("notebooks/tutorials/01_getting_started.ipynb")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(notebook_content)
    
    print("✅ 示例筆記本創建完成")

def create_gitignore():
    """創建.gitignore文件"""
    print("\n📝 創建.gitignore文件...")
    
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/saved_models/*
models/checkpoints/*
!models/saved_models/.gitkeep
!models/checkpoints/.gitkeep

# Logs
logs/*.log
*.log

# Results
results/backtests/*
results/reports/*
!results/backtests/.gitkeep
!results/reports/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# System
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local
.env.*.local

# Temporary files
*.tmp
*.temp
'''
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore文件創建完成")

def print_next_steps():
    """打印後續步驟"""
    next_steps = """
    
    🎉 設置完成！接下來的步驟：
    
    1️⃣  啟動虛擬環境：
       Windows: venv\\Scripts\\activate
       Linux/Mac: source venv/bin/activate
    
    2️⃣  安裝完整依賴包：
       pip install -r requirements.txt
    
    3️⃣  啟動Jupyter筆記本：
       jupyter lab
       
    4️⃣  開始學習教程：
       打開 notebooks/tutorials/01_getting_started.ipynb
    
    5️⃣  閱讀學習路線圖：
       查看 learning-roadmap.md
    
    📚 重要文件說明：
    - README.md: 主要學習資源和路線圖
    - learning-roadmap.md: 詳細12個月學習計劃
    - examples/: 實用代碼範例
    - config.toml: 項目配置文件
    
    💪 俄羅斯裁判的話：
    "準備工作已就緒，現在開始你的征程吧！
     記住：每一天的進步都在為你的專家之路奠定基礎！"
    
    🚀 祝你學習順利，成為頂尖的AI量化交易專家！
    """
    print(next_steps)

def main():
    """主函數"""
    print_banner()
    
    try:
        # 執行設置步驟
        check_python_version()
        check_system_requirements()
        
        # 詢問用戶是否繼續
        response = input("\n🤔 是否繼續設置學習環境？(y/n): ").lower().strip()
        if response not in ['y', 'yes', '是', '1']:
            print("👋 設置已取消，期待你的下次使用！")
            return
        
        create_virtual_environment()
        install_core_dependencies()
        setup_project_structure()
        create_sample_notebook()
        create_gitignore()
        
        print("\n" + "="*60)
        print("🎊 AI量化交易學習平台設置完成！")
        print("="*60)
        
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n👋 設置被用戶中斷")
    except Exception as e:
        print(f"\n❌ 設置過程中發生錯誤: {e}")
        print("請檢查錯誤信息並重試，或手動設置環境")

if __name__ == "__main__":
    main()
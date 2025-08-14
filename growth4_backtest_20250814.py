import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from datetime import datetime as dt

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("请先安装 yfinance ： pip install yfinance") from e

# ------------------ 配置参数 ------------------
TICKERS = ["TSLA", "NVDA", "PLTR", "HOOD"]
START_DATE = "2024-01-01"
END_DATE = dt.now().strftime("%Y-%m-%d")

# 用户实际购买记录 (示例格式)
USER_PURCHASES = {
    "TSLA": [
        {"date": "2025-08-14", "price": 339.40, "shares": 4}
    ],
    "NVDA": [
        {"date": "2028-08-13", "price": 182.99, "shares": 1},
        {"date": "2028-08-13", "price": 181.90, "shares": 1},
        {"date": "2028-08-14", "price": 182.57, "shares": 2},
    ],
    "PLTR": [
        {"date": "2028-08-14", "price": 189.00, "shares": 2},
        {"date": "2028-08-14", "price": 185.80, "shares": 2},
    ],
    "HOOD": [
        {"date": "2028-08-14", "price": 115.75, "shares": 5}
    ]
}

# 交易参数
DIP_PCT = 0.12   # 下跌触发加仓阈值
MAX_ADDS = 5      # 最大加仓次数
TP_PCT = 0.30     # 止盈阈值
SL_PCT = 0.20     # 止损阈值
REENTER_WHEN_FLAT = True
# ---------------------------------------------

def fetch_data(ticker: str,
               start: str = START_DATE,
               end: Optional[str] = END_DATE) -> pd.Series:
    # 下载单只股票的收盘价（已复权），尽量规避 yfinance 不同版本的列名结构差异。
    df = yf.download(
        ticker, start=start, end=end,
        auto_adjust=True, progress=False,
        group_by="column", threads=False
    )

    if df is None or df.empty:
        raise ValueError(f"{ticker}: 无法下载到价格数据")

    # 常见：单层列名，直接取 'Close'
    if "Close" in df.columns:
        s = df["Close"].copy()
    # 兼容：多层列名
    elif isinstance(df.columns, pd.MultiIndex):
        try:
            if "Close" in df.columns.get_level_values(0):
                s = df.xs("Close", axis=1, level=0)
            elif "Close" in df.columns.get_level_values(-1):
                s = df.xs("Close", axis=1, level=-1)
            else:
                candidates = [c for c in df.columns if isinstance(c, tuple) and any(str(x).lower()=="close" for x in c)]
                if not candidates:
                    raise KeyError("未找到 Close 列")
                s = df[candidates[0]]
        except Exception:
            s = df.iloc[:, 0]
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"].copy()
        else:
            s = df.iloc[:, 0]

    # 压成 Series（如果是 DataFrame，取第一列；如果是一列 DataFrame，用 squeeze）
    if isinstance(s, pd.DataFrame):
        if s.shape[1] >= 1:
            s = s.iloc[:, 0]
        s = s.squeeze()

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = ticker
    return s


def compute_stats(prices: pd.Series) -> dict:
    # 计算价格序列的年化收益、波动、最大回撤、夏普（rf≈0）
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = pd.to_numeric(prices, errors="coerce").dropna()

    daily = prices.pct_change().dropna()
    mean_val = float(daily.mean()) if len(daily) else np.nan
    std_val  = float(daily.std()) if len(daily) else np.nan

    # 计算年化收益率
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr = (prices.iloc[-1] / prices.iloc[0]) ** (1/years) - 1 if years > 0 else np.nan

    rolling_max = prices.cummax()
    drawdown = prices / rolling_max - 1.0
    # 加个层面的最大回撤
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan

    # 夏普比率，近似用无风险利率 0
    sharpe = (mean_val / std_val) * np.sqrt(252) if (std_val == std_val and std_val > 0) else np.nan
    # 年化波动率
    ann_vol = std_val * np.sqrt(252) if std_val == std_val else np.nan

    return {"CAGR": cagr, "Ann.Vol": ann_vol, "MaxDD": max_dd, "Sharpe": sharpe}


def calculate_actual_returns(ticker: str, purchases: List[Dict], current_price: float) -> Dict:
    """计算实际收益率"""
    total_cost = sum(p['price'] * p['shares'] for p in purchases)
    total_shares = sum(p['shares'] for p in purchases)
    current_value = current_price * total_shares
    absolute_return = current_value - total_cost
    percent_return = (current_value / total_cost - 1) * 100 if total_cost > 0 else 0
    
    return {
        "ticker": ticker,
        "total_shares": total_shares,
        "avg_cost": total_cost / total_shares if total_shares > 0 else 0,
        "current_price": current_price,
        "absolute_return": absolute_return,
        "percent_return": percent_return,
        "current_value": current_value
    }

def generate_trade_recommendation(ticker: str, 
                                purchases: List[Dict], 
                                current_price: float,
                                price_history: pd.Series) -> Dict:
    """生成交易建议"""
    if not purchases:
        return {"action": "NO_HOLDINGS", "reason": "未持有该股票"}
    
    # 计算平均成本
    total_shares = sum(p['shares'] for p in purchases)
    avg_cost = sum(p['price'] * p['shares'] for p in purchases) / total_shares
    
    # 当前盈亏状态
    current_return = (current_price / avg_cost - 1) * 100
    
    # 最近价格趋势 (20日均线)
    ma_20 = price_history.rolling(20).mean().iloc[-1]
    
    recommendations = []
    
    # 止盈建议
    if current_price >= avg_cost * (1 + TP_PCT):
        recommendations.append({
            "action": "SELL_PARTIAL",
            "reason": f"达到止盈点 {TP_PCT*100}% (当前回报: {current_return:.1f}%)",
            "suggested_shares": max(1, int(total_shares * 0.3))  # 建议卖出30%
        })
    
    # 止损建议
    elif current_price <= avg_cost * (1 - SL_PCT):
        recommendations.append({
            "action": "SELL_ALL",
            "reason": f"触发止损点 {SL_PCT*100}% (当前亏损: {abs(current_return):.1f}%)"
        })
    
    # 加仓建议 (基于最近低点)
    recent_low = price_history.rolling(10).min().iloc[-1]
    if current_price <= recent_low * (1 + DIP_PCT/2):
        dip_pct_from_low = (current_price - recent_low) / recent_low
        recommendations.append({
            "action": "BUY_MORE",
            "reason": f"接近近期低点 (低于{recent_low:.2f}, 差距{dip_pct_from_low*100:.1f}%)",
            "suggested_shares": max(1, int(total_shares * 0.2))  # 建议加仓20%
        })
    
    # 趋势建议
    if current_price > ma_20 * 1.05:
        recommendations.append({
            "action": "HOLD",
            "reason": f"价格高于20日均线{ma_20:.2f} (+{(current_price/ma_20-1)*100:.1f}%)"
        })
    elif current_price < ma_20 * 0.95:
        recommendations.append({
            "action": "WAIT",
            "reason": f"价格低于20日均线{ma_20:.2f} ({(current_price/ma_20-1)*100:.1f}%)"
        })
    
    return {
        "ticker": ticker,
        "current_price": current_price,
        "avg_cost": avg_cost,
        "current_return_pct": current_return,
        "recommendations": recommendations if recommendations else [{
            "action": "HOLD", 
            "reason": "无明确交易信号"
        }]
    }

def main():
    # 获取最新价格数据
    price_data = {t: fetch_data(t) for t in TICKERS}
    current_prices = {t: data.iloc[-1] for t, data in price_data.items()}
    
    # 计算实际收益率
    actual_returns = []
    for ticker in TICKERS:
        if ticker in USER_PURCHASES and USER_PURCHASES[ticker]:
            returns = calculate_actual_returns(
                ticker, 
                USER_PURCHASES[ticker], 
                current_prices[ticker]
            )
            actual_returns.append(returns)
    
    # 生成交易建议
    trade_recommendations = []
    for ticker in TICKERS:
        if ticker in USER_PURCHASES and USER_PURCHASES[ticker]:
            recommendation = generate_trade_recommendation(
                ticker,
                USER_PURCHASES[ticker],
                current_prices[ticker],
                price_data[ticker]
            )
            trade_recommendations.append(recommendation)
    
    # 打印结果
    print("\n=== 实际收益率 ===")
    returns_df = pd.DataFrame(actual_returns)
    print(returns_df.to_string(index=False))
    
    print("\n=== 交易建议 ===")
    for rec in trade_recommendations:
        print(f"\n股票: {rec['ticker']}")
        print(f"当前价: {rec['current_price']:.2f} | 平均成本: {rec['avg_cost']:.2f}")
        print(f"当前回报率: {rec['current_return_pct']:.1f}%")
        for i, action in enumerate(rec['recommendations'], 1):
            print(f"{i}. {action['action']}: {action['reason']}")
            if 'suggested_shares' in action:
                print(f"   建议数量: {action['suggested_shares']}股")

if __name__ == "__main__":
    main()
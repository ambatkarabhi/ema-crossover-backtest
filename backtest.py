# Backtest of the 50/200 EMA crossover strategy
# Invested when 50 EMA > 200 EMA, in cash otherwise.
#
# python backtest.py
# python backtest.py --ticker RELIANCE.NS --start 2005-01-01

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252
OUTPUT_DIR = Path("results")


# data
def load_prices(ticker: str, start: str, end: str | None = None) -> pd.Series:
    # daily close, adjusted for splits and dividends
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the ticker symbol.")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance sometimes returns a dataframe here
        close = close.iloc[:, 0]
    return close.dropna().rename("close")


# strategy
def run_strategy(close: pd.Series, fast: int = 50, slow: int = 200,
                 cost_pct: float = 0.1) -> pd.DataFrame:
    # cost_pct = cost per buy or sell in %
    df = close.to_frame()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()

    # 1 = invested, 0 = cash. skip the first 200 days so the slow EMA settles
    df["signal"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
    df = df.iloc[slow:].copy()

    # take the trade the next day to avoid look-ahead bias
    df["position"] = df["signal"].shift(1).fillna(0)

    df["market_ret"] = df["close"].pct_change().fillna(0)
    trades = df["position"].diff().abs().fillna(0)          # 1 whenever we buy or sell
    df["strategy_ret"] = df["position"] * df["market_ret"] - trades * cost_pct / 100

    df["equity_strategy"] = (1 + df["strategy_ret"]).cumprod()
    df["equity_buyhold"] = (1 + df["market_ret"]).cumprod()
    return df


# performance metrics
def max_drawdown(equity: pd.Series) -> float:
    return (equity / equity.cummax() - 1).min()


def list_trades(df: pd.DataFrame) -> pd.DataFrame:
    # one row per trade
    pos = df["position"]
    entries = df.index[(pos == 1) & (pos.shift(1, fill_value=0) == 0)]
    exits = df.index[(pos == 0) & (pos.shift(1, fill_value=0) == 1)]

    rows = []
    for entry in entries:
        later_exits = exits[exits > entry]
        exit_ = later_exits[0] if len(later_exits) else df.index[-1]
        held = df.loc[entry:exit_]
        # don't count the exit day itself
        rets = held["strategy_ret"].iloc[:-1] if len(later_exits) else held["strategy_ret"]
        rows.append({
            "entry": entry.date(),
            "exit": exit_.date(),
            "days": (exit_ - entry).days,
            "return_pct": ((1 + rets).prod() - 1) * 100,
            "status": "closed" if len(later_exits) else "open",
        })
    return pd.DataFrame(rows)


def summarise(returns: pd.Series, equity: pd.Series) -> dict:
    years = len(returns) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS) if returns.std() else np.nan
    return {
        "Total return": f"{(equity.iloc[-1] - 1) * 100:,.1f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Annual volatility": f"{vol * 100:.2f}%",
        "Sharpe ratio (rf=0)": f"{sharpe:.2f}",
        "Max drawdown": f"{max_drawdown(equity) * 100:.2f}%",
    }


# charts
def plot_signals(df: pd.DataFrame, ticker: str, path: Path) -> None:
    pos = df["position"]
    buys = df.index[(pos == 1) & (pos.shift(1, fill_value=0) == 0)]
    sells = df.index[(pos == 0) & (pos.shift(1, fill_value=0) == 1)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["close"], color="#9aa0a6", lw=1, label="Close")
    ax.plot(df.index, df["ema_fast"], color="#1a73e8", lw=1.4, label="50 EMA")
    ax.plot(df.index, df["ema_slow"], color="#e8710a", lw=1.4, label="200 EMA")
    ax.scatter(buys, df.loc[buys, "close"], marker="^", color="#188038", s=80, zorder=3, label="Buy (golden cross)")
    ax.scatter(sells, df.loc[sells, "close"], marker="v", color="#d93025", s=80, zorder=3, label="Sell (death cross)")
    ax.set_title(f"{ticker}: 50/200 EMA crossover signals")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity(df: pd.DataFrame, ticker: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df.index, df["equity_strategy"], color="#1a73e8", lw=1.6, label="EMA strategy")
    ax1.plot(df.index, df["equity_buyhold"], color="#9aa0a6", lw=1.4, label="Buy & hold")
    ax1.set_title(f"{ticker}: growth of 1 invested")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    for col, color, label in [("equity_strategy", "#1a73e8", "EMA strategy"),
                              ("equity_buyhold", "#9aa0a6", "Buy & hold")]:
        dd = (df[col] / df[col].cummax() - 1) * 100
        ax2.plot(df.index, dd, color=color, lw=1.2, label=label)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# main
def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a 50/200 EMA crossover strategy.")
    parser.add_argument("--ticker", default="^NSEI", help="Yahoo Finance ticker (default: ^NSEI = Nifty 50)")
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--fast", type=int, default=50, help="Fast EMA period")
    parser.add_argument("--slow", type=int, default=200, help="Slow EMA period")
    parser.add_argument("--cost", type=float, default=0.1, help="Cost per trade side, in percent")
    args = parser.parse_args()

    close = load_prices(args.ticker, args.start, args.end)
    df = run_strategy(close, args.fast, args.slow, args.cost)
    trades = list_trades(df)

    strat = summarise(df["strategy_ret"], df["equity_strategy"])
    bh = summarise(df["market_ret"], df["equity_buyhold"])
    summary = pd.DataFrame({"EMA strategy": strat, "Buy & hold": bh})

    closed = trades[trades["status"] == "closed"] if not trades.empty else trades
    win_rate = (closed["return_pct"] > 0).mean() * 100 if len(closed) else float("nan")
    time_in_market = df["position"].mean() * 100

    print(f"\n{args.ticker} | {df.index[0].date()} to {df.index[-1].date()} | "
          f"{args.fast}/{args.slow} EMA | cost {args.cost}% per side\n")
    print(summary.to_string())
    print(f"\nNumber of trades: {len(trades)}   Win rate: {win_rate:.0f}%   "
          f"Time in market: {time_in_market:.0f}%\n")
    print(trades.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    OUTPUT_DIR.mkdir(exist_ok=True)
    name = args.ticker.replace("^", "").replace(".", "_")
    plot_signals(df, args.ticker, OUTPUT_DIR / f"{name}_signals.png")
    plot_equity(df, args.ticker, OUTPUT_DIR / f"{name}_equity.png")
    summary.to_csv(OUTPUT_DIR / f"{name}_summary.csv")
    trades.to_csv(OUTPUT_DIR / f"{name}_trades.csv", index=False)
    print(f"\nCharts and CSVs saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

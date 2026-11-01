# EMA Crossover Backtest (50/200) on Nifty 50

I wanted to check whether the classic 50/200 moving average crossover actually works on the Indian market, so I backtested it on the Nifty 50 and compared it with simply buying and holding the index.

The rule is simple:

- When the 50 day EMA moves above the 200 day EMA (golden cross), buy and stay invested.
- When the 50 day EMA moves below the 200 day EMA (death cross), sell and stay in cash.

I used exponential moving averages instead of simple ones because they put more weight on recent prices and react a bit faster when the trend changes.

## How the backtest works

- Prices are daily closing prices from Yahoo Finance (adjusted for splits and dividends).
- The signal is calculated on the day's close but the trade is taken the next day. This avoids using information that wouldn't have been available at the time.
- Every buy and sell is charged 0.1% to account for brokerage, taxes and slippage. This can be changed with `--cost`.
- The first 200 days are skipped because the 200 EMA needs enough history before it means anything.
- The strategy is long only. It doesn't short, and the cash earns nothing while out of the market.

## What it measures

- Total return and CAGR
- Annual volatility
- Sharpe ratio (risk free rate taken as zero)
- Maximum drawdown, i.e. the biggest fall from a peak
- Number of trades, win rate and percentage of time invested

All of these are shown for both the strategy and buy & hold over the same period.

## Results

<!-- add summary table here after running -->

![Signals](results/NSEI_signals.png)
![Equity curve](results/NSEI_equity.png)

## Running it

```bash
git clone https://github.com/ambatkarabhi/ema-crossover-backtest.git
cd ema-crossover-backtest
pip install -r requirements.txt

python backtest.py
```

Some other examples:

```bash
python backtest.py --ticker RELIANCE.NS
python backtest.py --start 2005-01-01 --cost 0.15
python backtest.py --fast 20 --slow 100
```

The charts and CSV files (summary and list of trades) are saved in the `results` folder.

To run it in Google Colab, use this in a cell:

```
!git clone https://github.com/ambatkarabhi/ema-crossover-backtest.git
%cd ema-crossover-backtest
!pip install -q -r requirements.txt
!python backtest.py
```

## Files

- `backtest.py` - downloads the data, runs the strategy, calculates the metrics and makes the charts
- `requirements.txt` - Python packages needed
- `results/` - output charts and CSVs

## Limitations

- Only tested on one index for one period, so the results may not hold elsewhere.
- Trades are assumed at the closing price. Actual prices could be different, especially when the market gaps.
- Moving averages are lagging indicators, so the strategy enters and exits late and can get whipsawed when the market moves sideways.
- No interest is added on cash.

## Things I want to add

- Interest on idle cash (FD rate) as a fairer comparison
- Testing on other indices and individual stocks
- Trying different EMA lengths with walk forward testing so the results aren't overfitted

Data is from Yahoo Finance using the `yfinance` library. This is a learning project and not investment advice.

"""
Price data + returns for the portfolio.

Pulls daily adjusted close from Yahoo, builds per-name and portfolio
daily return series. Simple returns rather than log: portfolio return
is then just sum_i(w_i * r_i), which keeps the VaR step linear.

For the 1d/5d horizons we care about, simple vs log diverge by < 0.1bp,
so the choice is mostly about downstream convenience.
"""

from datetime import date
import pandas as pd
import yfinance as yf

from portfolio import tickers, weight

# 2007 start so the GFC stress window has full pre-crash history
START_DATE = "2007-01-01"


def fetch_prices(start=START_DATE, end=None):
    if end is None:
        end = date.today().isoformat()
    df = yf.download(tickers(), start=start, end=end,
                     auto_adjust=True, progress=False)
    # multi-ticker download gives MultiIndex columns (field, ticker),
    # we just want Close
    return df["Close"]


def daily_returns(prices):
    return prices.pct_change().dropna()


def portfolio_returns(asset_returns):
    """r_p = sum_i(w_i * r_i), with weights aligned to the columns."""
    w = pd.Series({t: weight(t) for t in asset_returns.columns})
    return asset_returns @ w


if __name__ == "__main__":
    print("fetching from yahoo...")
    prices = fetch_prices()
    print(f"got {prices.shape[0]} trading days, {prices.shape[1]} tickers")
    print(f"range: {prices.index[0].date()} to {prices.index[-1].date()}\n")

    rets = daily_returns(prices)
    port = portfolio_returns(rets)

    # quick sanity check - if vol comes out way off historical norms
    # something's wrong with the data
    print("portfolio return stats:")
    print(f"  ann. mean:  {port.mean() * 252:>7.2%}")
    print(f"  ann. vol:   {port.std() * 252**0.5:>7.2%}")
    print(f"  best day:   {port.max():>7.2%}  on {port.idxmax().date()}")
    print(f"  worst day:  {port.min():>7.2%}  on {port.idxmin().date()}")
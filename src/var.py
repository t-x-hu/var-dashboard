"""
Historical Simulation VaR and ES.

99% confidence by default. 1-day uses raw daily returns; 5-day uses
rolling 5-day cumulative returns instead of sqrt-time scaling, which
captures actual serial dependence in the data. Vol clustering means
sqrt(5) tends to understate 5-day tail risk in stressed regimes (the
GFC and COVID windows in our data make this point fairly directly).
"""

import numpy as np

CONFIDENCE = 0.99


def historical_var(returns, confidence=CONFIDENCE):
    """
    -quantile_alpha(returns), where alpha = 1 - confidence.
    Returned as positive loss magnitude (so 0.029 means 2.9% loss).
    """
    return -np.quantile(returns, 1 - confidence)


def historical_es(returns, confidence=CONFIDENCE):
    """E[L | L >= VaR_alpha], average loss in the tail beyond VaR."""
    threshold = np.quantile(returns, 1 - confidence)
    return -returns[returns <= threshold].mean()


def rolling_returns(returns, window):
    return (1 + returns).rolling(window).apply(np.prod) - 1


if __name__ == "__main__":
    from returns import fetch_prices, daily_returns, portfolio_returns

    print("loading returns...")
    prices = fetch_prices()
    rets = daily_returns(prices)
    port = portfolio_returns(rets)
    print(f"loaded {len(port)} days of portfolio returns\n")

    print("99% Historical Simulation:")
    var_1d = historical_var(port)
    es_1d = historical_es(port)
    print(f"  1-day  VaR  {var_1d:>6.2%}   ES  {es_1d:>6.2%}")

    port_5d = rolling_returns(port, 5).dropna()
    var_5d = historical_var(port_5d)
    es_5d = historical_es(port_5d)
    print(f"  5-day  VaR  {var_5d:>6.2%}   ES  {es_5d:>6.2%}")
    port_10d = rolling_returns(port, 10).dropna()
    var_10d = historical_var(port_10d)
    es_10d = historical_es(port_10d)
    print(f"  10-day VaR  {var_10d:>6.2%}   ES  {es_10d:>6.2%}")

    # how much extra does ES capture vs VaR?
    print(f"\n  ES/VaR ratio (1d): {es_1d/var_1d:.2f}x")
    # sqrt-time scaling sanity check
    scaled = var_1d * 5**0.5
    print(f"  sqrt(5)*1d_VaR = {scaled:.2%} vs actual 5d_VaR = {var_5d:.2%}")
"""
Component Expected Shortfall attribution by sector (Euler allocation).

ES_P = -E[X_P | X_P <= -VaR_P] is positively homogeneous in weights,
so it admits Euler decomposition: ES_P = sum_i CES_i, where
CES_i = -w_i * E[X_i | X_P <= -VaR_P]. Aggregating CES over assets in
a sector gives the sector's contribution to portfolio tail risk.

The output to look at is risk_share vs capital_share per sector. When
tail-risk share materially exceeds capital share, that's concentration
the weight column alone wouldn't surface.
"""

import pandas as pd

from portfolio import HOLDINGS, sector
from var import historical_var, historical_es

CONFIDENCE = 0.99


def asset_ces(asset_returns, portfolio_returns, confidence=CONFIDENCE):
    """
    CES_i = -w_i * E[X_i | X_P <= -VaR_P], asset i's negative mean return
    on portfolio-tail days, scaled by weight. Returns Series indexed by
    ticker as positive loss magnitudes.
    """
    var_p = historical_var(portfolio_returns, confidence)
    tail = portfolio_returns <= -var_p
    weights = pd.Series({t: HOLDINGS[t][1] for t in asset_returns.columns})
    return -asset_returns[tail].mean() * weights


def sector_ces(ces):
    """Aggregate asset-level CES to sector totals."""
    sectors = pd.Series({t: sector(t) for t in ces.index})
    return ces.groupby(sectors).sum()


if __name__ == "__main__":
    from returns import fetch_prices, daily_returns, portfolio_returns

    print("loading returns...")
    prices = fetch_prices()
    rets = daily_returns(prices)
    port = portfolio_returns(rets)
    print(f"loaded {len(port)} days\n")

    es_p = historical_es(port)
    asset_c = asset_ces(rets, port)
    sec_c = sector_ces(asset_c)
    recon = sec_c.sum()

    print(f"Portfolio 99% 1-day ES: {es_p:.4%}")
    print(f"Sum of sector CES:      {recon:.4%}   (diff {recon - es_p:+.4%})\n")

    weights = pd.Series({t: HOLDINGS[t][1] for t in HOLDINGS})
    sectors = pd.Series({t: HOLDINGS[t][0] for t in HOLDINGS})
    sec_weight = weights.groupby(sectors).sum()

    print(f"{'sector':<24} {'cap':>6} {'CES':>7} {'risk':>7} {'diff':>7}")
    print("-" * 56)
    for s in sec_c.sort_values(ascending=False).index:
        cap = sec_weight[s]
        ces = sec_c[s]
        risk = ces / recon
        print(f"{s:<24} {cap:>6.1%} {ces:>7.2%} {risk:>7.1%} {risk - cap:>+7.1%}")
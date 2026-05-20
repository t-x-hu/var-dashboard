"""
Monte Carlo VaR and ES under a normal assumption.

Fit (mu, sigma) to historical portfolio returns, draw 10k IID gaussian
samples, run the same VaR/ES math as var.py. 5-day horizon is sum of
5 IID daily draws.

The normal assumption is wrong on purpose. It's the standard parametric
baseline, and the whole point is the gap vs the historical sim - that
gap is what FRTB cared about when it moved the regulatory metric off
VaR onto ES in 2016.
"""

import numpy as np

from var import historical_var, historical_es

SIMS = 10_000
SEED = 42  # fix seed so the comparison vs historical is stable across runs


def mc_returns_normal(historical_returns, n_sims=SIMS, horizon=1, seed=SEED):
    """
    n_sims draws from N(mu, sigma) fit to history.
    horizon > 1 returns are sums of IID daily draws.
    """
    mu = historical_returns.mean()
    sigma = historical_returns.std()
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=(n_sims, horizon)).sum(axis=1)


def mc_var(historical_returns, confidence=0.99, n_sims=SIMS, horizon=1, seed=SEED):
    sims = mc_returns_normal(historical_returns, n_sims, horizon, seed)
    return historical_var(sims, confidence)


def mc_es(historical_returns, confidence=0.99, n_sims=SIMS, horizon=1, seed=SEED):
    sims = mc_returns_normal(historical_returns, n_sims, horizon, seed)
    return historical_es(sims, confidence)


if __name__ == "__main__":
    from returns import fetch_prices, daily_returns, portfolio_returns

    print("loading returns...")
    prices = fetch_prices()
    rets = daily_returns(prices)
    port = portfolio_returns(rets)
    print(f"loaded {len(port)} days")
    print(f"mu = {port.mean():.4%}/day, sigma = {port.std():.4%}/day\n")

    print(f"99% Monte Carlo ({SIMS:,} sims, normal):")
    mc_var_1d = mc_var(port)
    mc_es_1d = mc_es(port)
    print(f"  1-day  VaR  {mc_var_1d:>6.2%}   ES  {mc_es_1d:>6.2%}")

    mc_var_5d = mc_var(port, horizon=5)
    mc_es_5d = mc_es(port, horizon=5)
    print(f"  5-day  VaR  {mc_var_5d:>6.2%}   ES  {mc_es_5d:>6.2%}")

    # how much tail does the normal MC miss vs historical?
    hist_var_1d = historical_var(port)
    hist_es_1d = historical_es(port)
    var_gap = (hist_var_1d - mc_var_1d) / hist_var_1d
    es_gap = (hist_es_1d - mc_es_1d) / hist_es_1d
    print(f"\n  vs Historical (1-day):")
    print(f"  VaR  hist {hist_var_1d:.2%} - MC {mc_var_1d:.2%} = {var_gap:>+6.1%} gap")
    print(f"  ES   hist {hist_es_1d:.2%} - MC {mc_es_1d:.2%} = {es_gap:>+6.1%} gap")
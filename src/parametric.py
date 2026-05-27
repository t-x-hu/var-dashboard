"""
Parametric (variance-covariance) VaR and ES under a normal assumption.

Same N(mu, sigma) fit as monte_carlo.py, but VaR comes from the
inverse normal CDF and ES from the truncated-normal mean rather than
empirical quantiles of simulated draws. The two should match up to
MC sampling noise - if they diverge by more than ~5bp on 10k sims,
something's off in one of them.
"""

import numpy as np
from scipy.stats import norm

from var import historical_var, historical_es

CONFIDENCE = 0.99


def parametric_var(returns, confidence=CONFIDENCE, horizon=1):
    """
    -mu_h + sigma_h * z, where mu_h = horizon * mu, sigma_h = sqrt(horizon) * sigma,
    z = norm.ppf(confidence) (~2.326 at 99%). Sqrt-time scaling assumes IID -
    intentionally different from var.py's rolling windows for the comparison.
    Returned as positive loss magnitude.
    """
    mu = returns.mean()
    sigma = returns.std()
    mu_h = horizon * mu
    sigma_h = np.sqrt(horizon) * sigma
    z = norm.ppf(confidence)
    return -mu_h + z * sigma_h


def parametric_es(returns, confidence=CONFIDENCE, horizon=1):
    """
    Closed-form E[L | L >= VaR_alpha] under N(mu, sigma):
    -mu_h + sigma_h * phi(z) / alpha, with phi the standard normal pdf,
    z = norm.ppf(confidence), alpha = 1 - confidence.
    """
    mu = returns.mean()
    sigma = returns.std()
    mu_h = horizon * mu
    sigma_h = np.sqrt(horizon) * sigma
    z = norm.ppf(confidence)
    alpha = 1 - confidence
    return -mu_h + sigma_h * norm.pdf(z) / alpha


if __name__ == "__main__":
    from returns import fetch_prices, daily_returns, portfolio_returns

    print("loading returns...")
    prices = fetch_prices()
    rets = daily_returns(prices)
    port = portfolio_returns(rets)
    print(f"loaded {len(port)} days\n")

    print("99% Parametric (normal):")
    for h in [1, 5, 10]:
        var_p = parametric_var(port, horizon=h)
        es_p = parametric_es(port, horizon=h)
        print(f"  {h:>2}-day  VaR  {var_p:>6.2%}   ES  {es_p:>6.2%}")

    # param vs MC normal - identical up to sampling
    from monte_carlo import mc_var, mc_es
    param_var_1d = parametric_var(port)
    param_es_1d = parametric_es(port)
    mc_var_1d = mc_var(port)
    mc_es_1d = mc_es(port)
    print(f"\n  vs MC normal (1-day):")
    print(f"  VaR  param {param_var_1d:.2%}  MC {mc_var_1d:.2%}")
    print(f"  ES   param {param_es_1d:.2%}  MC {mc_es_1d:.2%}")

    # param vs historical - fat tail gap
    hist_var_1d = historical_var(port)
    hist_es_1d = historical_es(port)
    var_gap = (hist_var_1d - param_var_1d) / hist_var_1d
    es_gap = (hist_es_1d - param_es_1d) / hist_es_1d
    print(f"\n  vs Historical (1-day):")
    print(f"  VaR  hist {hist_var_1d:.2%} - param {param_var_1d:.2%} = {var_gap:>+6.1%} gap")
    print(f"  ES   hist {hist_es_1d:.2%} - param {param_es_1d:.2%} = {es_gap:>+6.1%} gap")
"""
VaR backtesting: Kupiec POF (unconditional coverage) and Christoffersen
independence (Markov). Both are LR statistics asymptotically chi-square(1)
under the null; combined as conditional coverage ~ chi-square(2).

Rolling 252-day estimation window, one-day-ahead forecast compared
against the realized next-day return. The exceedance series is the
binary input the tests operate on.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

CONFIDENCE = 0.99
WINDOW = 252  # one trading year, Basel standard


def rolling_var(returns, var_func, window=WINDOW, confidence=CONFIDENCE):
    """
    Walks forward: VaR_t from returns[t-window:t], paired with realized
    return[t]. Returns DataFrame ['var', 'realized'] indexed by date,
    with 'var' as positive loss magnitude.
    """
    vars_ = []
    for t in range(window, len(returns)):
        est = returns.iloc[t-window:t]
        vars_.append(var_func(est, confidence=confidence))
    return pd.DataFrame({
        "var": vars_,
        "realized": returns.iloc[window:].values,
    }, index=returns.index[window:])


def exceedances(bt):
    """1 where realized loss exceeds VaR, 0 otherwise."""
    return (bt["realized"] < -bt["var"]).astype(int).values


def kupiec_pof(exceed, alpha=1-CONFIDENCE):
    """
    LR_POF = -2 ln[ p^N (1-p)^(T-N) / (N/T)^N (1-N/T)^(T-N) ],
    p = alpha, N = sum(exceed), T = len(exceed). ~chi2(1) under H0.
    """
    T = len(exceed)
    N = int(exceed.sum())
    if N == 0 or N == T:
        return np.nan, np.nan
    pi_hat = N / T
    log_L0 = N * np.log(alpha) + (T - N) * np.log(1 - alpha)
    log_L1 = N * np.log(pi_hat) + (T - N) * np.log(1 - pi_hat)
    LR = -2 * (log_L0 - log_L1)
    return LR, 1 - chi2.cdf(LR, df=1)


def christoffersen_ind(exceed):
    """
    Markov-chain test: H0: pi_01 = pi_11 = pi (no clustering).
    Builds 2x2 transition counts and tests equality of conditional
    exceedance probabilities. ~chi2(1) under H0.
    """
    prev, curr = exceed[:-1], exceed[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    if pi in (0, 1) or pi_01 in (0, 1) or pi_11 in (0, 1):
        return np.nan, np.nan

    log_L0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    log_L1 = (n00 * np.log(1 - pi_01) + n01 * np.log(pi_01)
              + n10 * np.log(1 - pi_11) + n11 * np.log(pi_11))
    LR = -2 * (log_L0 - log_L1)
    return LR, 1 - chi2.cdf(LR, df=1)


def conditional_coverage(exceed, alpha=1-CONFIDENCE):
    """LR_CC = LR_POF + LR_IND ~ chi2(2) under joint H0."""
    LR_pof, _ = kupiec_pof(exceed, alpha)
    LR_ind, _ = christoffersen_ind(exceed)
    if np.isnan(LR_pof) or np.isnan(LR_ind):
        return np.nan, np.nan
    LR = LR_pof + LR_ind
    return LR, 1 - chi2.cdf(LR, df=2)


if __name__ == "__main__":
    from returns import fetch_prices, daily_returns, portfolio_returns
    from var import historical_var
    from parametric import parametric_var
    from monte_carlo import mc_var

    print("loading returns...")
    prices = fetch_prices()
    rets = daily_returns(prices)
    port = portfolio_returns(rets)
    print(f"loaded {len(port)} days\n")

    methods = {
        "Historical": historical_var,
        "Parametric": parametric_var,
        "MC normal":  mc_var,
    }

    print(f"Rolling {WINDOW}-day backtest, 99% 1-day VaR.")
    print(f"{'method':<12} {'T':>5} {'N':>4} {'exp':>5}  {'LR_POF':>7} {'LR_IND':>7} {'LR_CC':>7}  verdict")
    print("-" * 72)

    chi2_cc_95 = chi2.ppf(0.95, df=2)
    for name, fn in methods.items():
        bt = rolling_var(port, fn)
        exc = exceedances(bt)
        T, N = len(exc), int(exc.sum())
        expected = T * (1 - CONFIDENCE)
        LR_pof, _ = kupiec_pof(exc)
        LR_ind, _ = christoffersen_ind(exc)
        LR_cc, _ = conditional_coverage(exc)
        verdict = "reject" if LR_cc > chi2_cc_95 else "pass"
        print(f"{name:<12} {T:>5} {N:>4} {expected:>5.1f}  "
              f"{LR_pof:>7.2f} {LR_ind:>7.2f} {LR_cc:>7.2f}  {verdict}")
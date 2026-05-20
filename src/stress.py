"""
Stress testing against three crisis windows: 2008 GFC, March 2020 COVID,
and the 2022 rate-hike regime. For each window we report worst single-day,
worst 5-day, and peak-to-trough drawdown, and express the 5-day stress
as a multiple of the model VaR.

Point of the exercise: a 99% 1-day or 5-day VaR is a probabilistic
statement about normal days. Real crises sit outside the model's
distribution. The ratio "stress / VaR" quantifies how far outside.
"""

from var import historical_var, rolling_returns

# acute phase of each crisis, not the full event window
WINDOWS = {
    "2008 GFC":         ("2008-09-01", "2008-12-31"),
    "March 2020 COVID": ("2020-02-19", "2020-03-23"),
    "2022 rate hikes":  ("2022-01-01", "2022-12-31"),
}


def stress_stats(portfolio_returns, start, end):
    """worst day, worst rolling 5d, peak-to-trough drawdown for one window."""
    window = portfolio_returns.loc[start:end]
    worst_1d = window.min()
    worst_5d = rolling_returns(window, 5).min()
    # drawdown = current level vs running peak, take the lowest point
    cum = (1 + window).cumprod()
    drawdown = (cum / cum.cummax() - 1).min()
    return {
        "n_days": len(window),
        "worst_1d": worst_1d,
        "worst_5d": worst_5d,
        "drawdown": drawdown,
    }


if __name__ == "__main__":
    from returns import fetch_prices, daily_returns, portfolio_returns

    print("loading returns...")
    prices = fetch_prices()
    rets = daily_returns(prices)
    port = portfolio_returns(rets)
    print(f"loaded {len(port)} days\n")

    # model VaR for the comparison ratio
    var_1d = historical_var(port)
    port_5d = rolling_returns(port, 5).dropna()
    var_5d = historical_var(port_5d)
    print(f"Model VaR (99% historical): 1-day {var_1d:.2%}, 5-day {var_5d:.2%}\n")

    print(f"{'window':<20} {'days':>5} {'worst 1d':>10} {'worst 5d':>10} "
          f"{'drawdown':>10} {'5d/VaR':>8}")
    print("-" * 68)
    for name, (start, end) in WINDOWS.items():
        s = stress_stats(port, start, end)
        # worst_5d is signed (negative for loss); VaR is unsigned magnitude
        ratio = abs(s["worst_5d"]) / var_5d
        print(f"{name:<20} {s['n_days']:>5} {s['worst_1d']:>10.2%} "
              f"{s['worst_5d']:>10.2%} {s['drawdown']:>10.2%} {ratio:>7.1f}x")
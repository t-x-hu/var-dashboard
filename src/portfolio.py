"""
Portfolio definition for the VaR/ES dashboard.

15 large-cap US names across 6 GICS sectors. Weights are a rough match
to S&P 500 sector weights (top 6 sectors, ~80% of the index).
Total notional: $10mm.

All tickers have clean daily history back through 2007, so the 2008
GFC stress scenario has real data to draw from.
"""

NOTIONAL = 10_000_000  # USD

# ticker -> (sector, weight)
HOLDINGS = {
    # Tech (33%)
    "AAPL":  ("Information Technology",     0.10),
    "MSFT":  ("Information Technology",     0.10),
    "NVDA":  ("Information Technology",     0.05),
    "GOOGL": ("Information Technology",     0.05),
    "ORCL":  ("Information Technology",     0.03),
    # Financials (20%) - 2008 was brutal here, watch the stress test
    "JPM":   ("Financials",                 0.08),
    "BAC":   ("Financials",                 0.06),
    "GS":    ("Financials",                 0.06),
    # Healthcare (20%)
    "JNJ":   ("Health Care",                0.08),
    "UNH":   ("Health Care",                0.07),
    "PFE":   ("Health Care",                0.05),
    # Consumer Disc (13%)
    "AMZN":  ("Consumer Discretionary",     0.08),
    "HD":    ("Consumer Discretionary",     0.05),
    # Industrials (7%) - CAT is also my credit-research project, nice cross-reference
    "CAT":   ("Industrials",                0.07),
    # Energy (7%)
    "XOM":   ("Energy",                     0.07),
}

# weights must sum to 1
_total = sum(w for _, w in HOLDINGS.values())
assert abs(_total - 1.0) < 1e-9, f"weights sum to {_total}"


def tickers():
    return list(HOLDINGS.keys())

def weight(t):
    return HOLDINGS[t][1]

def sector(t):
    return HOLDINGS[t][0]

def dollar_position(t):
    """Dollar position size at $10mm total notional."""
    return NOTIONAL * weight(t)


if __name__ == "__main__":
    print(f"Portfolio: {len(tickers())} tickers, ${NOTIONAL:,.0f} notional")
    for t in tickers():
        print(f"  {t:6s} {sector(t):24s} {weight(t)*100:5.1f}%  ${dollar_position(t):>12,.0f}")
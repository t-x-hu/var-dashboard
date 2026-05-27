# VaR / ES Risk Engine

99% Value-at-Risk and Expected Shortfall computed three ways (Historical Simulation, Parametric variance-covariance, and Monte Carlo) on a 15-name $10mm US equity portfolio over 4,877 trading days (2007–present), with rolling backtests, historical stress scenarios, and sector-level component ES attribution. Methodology aligned with Basel III FRTB Internal Models Approach.

## Findings

1. **Fat-tail signature at 1-day**. Historical ES 5.70% vs Monte Carlo normal ES 3.71% at 99%; Hist/MC ratio of 1.54x, the magnitude normal-fit misses. Excess kurtosis of daily returns ≈ 12.7.

2. **Gap collapses over horizon**. 10-day Hist/MC ES ratio = 1.39x (15.12% / 10.90%), down from 1.54x at 1-day. Consistent with CLT damping fat-tail effect as horizon grows. Empirical motivation for FRTB requiring stressed-window calibration rather than horizon-scaling the full-sample VaR.

3. **All three methods reject conditional coverage**. Rolling 252-day backtest over 4,625 windows. Parametric and MC fail Kupiec POF hard (LR 88 / 71 vs 3.84 critical) from fat-tail underestimation. Historical's POF failure is milder (LR 19) but still rejects, and also fails Christoffersen independence (LR 9.3) on exceedance clustering — vol regimes drive correlated breaches. LR_CC ranges 28–98, all reject at 95%.

4. **Crisis stress = 1.3–2.5x model VaR**. Worst 5-day returns: GFC 2008 = −20.2% (2.5x), March 2020 COVID = −18.1% (2.2x), 2022 rate cycle = −10.3% (1.3x). Five-day 99% historical VaR = 8.06%.

5. **Tail risk concentrated in Financials, not Tech**. Component ES decomposition by sector. Financials 20% capital → 30.1% tail risk share (+10.1pp over-contribution), reflecting GFC-era stress on JPM/BAC/GS. Information Technology, despite 33% capital, near-parity at 30.0% on risk: mega-cap franchises (AAPL/MSFT/NVDA) dampen left tails relative to banks. Health Care 20% capital → 14.5% risk (−5.5pp), defensive as expected.

## Run

```bash
pip install -r requirements.txt
python src/var.py           # Historical VaR + ES, 1d / 5d / 10d
python src/parametric.py    # Closed-form variance-covariance, cross-checked vs MC
python src/monte_carlo.py   # MC normal VaR + ES, gap vs Historical
python src/stress.py        # GFC / COVID / 2022 crisis windows
python src/backtest.py      # Kupiec POF + Christoffersen + CC, three methods
python src/attribution.py   # Component ES by sector (Euler decomposition)
```

## Project Structure

```
src/
  portfolio.py     15-name US equity portfolio, 6 GICS sectors, $10mm notional
  returns.py       yfinance daily price fetch + return calculation
  var.py           Historical Simulation VaR + ES
  parametric.py    Parametric VaR + ES (normal closed-form)
  monte_carlo.py   Monte Carlo normal VaR + ES (10k sims, fixed seed)
  stress.py        Three historical crisis windows
  backtest.py      Kupiec POF + Christoffersen independence + CC
  attribution.py   Component ES by sector (Euler decomposition)
  storage.py       SQLite persistence: positions, returns, risk_runs
```

## Limitations

1. **No conditional vol modeling**. All three methods assume IID returns; the Christoffersen test confirms IID fails empirically. GARCH(1,1) filtering (VaR on filtered innovations) is the next step.

2. **Normal-fit MC adds nothing over Parametric**. Same distribution + fixed seed → essentially identical VaR/ES (modulo sampling noise). MC's actual value (heavy tails, jumps, multi-asset correlation) is untapped here. Extension to t-distribution MC or empirical bootstrap would actually exercise MC's flexibility.

3. **Backtest uses rolling 252-day window**. FRTB IMA actually requires calibration on a fixed stressed window (e.g., 2008–2009). Current implementation tracks current-regime VaR, not stressed-period VaR.

4. **Single asset class**. US equity only. Multi-asset extension would require correlation matrix modeling and Cholesky factorization for MC.
# Multi-Asset Portfolio VaR / ES Dashboard

Daily and 5-day 99% Value-at-Risk and Expected Shortfall on a 15-ticker US equity
portfolio ($10mm notional), via Historical Simulation and Monte Carlo (normal).
Stress-tested against 2008 GFC, March 2020 COVID, and 2022 rate hikes. Full
narrative in `notebooks/analysis.ipynb`.

Why I built this: a practical comparison of the VaR vs ES distinction behind the
Basel III FRTB 2016 shift to stressed Expected Shortfall. The same fat-tail
mechanism shows up in counterparty credit risk, which I cover in a separate
Caterpillar (CAT) credit research project.

## Headline numbers

|                       | 1-day VaR | 1-day ES | 5-day VaR | 5-day ES |
|-----------------------|-----------|----------|-----------|----------|
| Historical            | 3.89%     | 5.70%    | 8.06%     | 11.33%   |
| Monte Carlo (normal)  | 3.23%     | 3.71%    | 6.82%     | 7.70%    |
| Hist / MC ratio       | 1.21x     | 1.54x    | 1.18x     | 1.47x    |

In dollars on $10mm notional, 1-day 99% ES: Historical $570k vs MC normal $371k.
The 53% gap on ES is the empirical evidence behind the FRTB shift from VaR to
stressed ES. Normal-fit MC systematically misses tail mass that Historical
Simulation captures by construction.

## Crisis stress tests

Worst observed 5-day losses vs the model 5-day 99% VaR (8.06%):

- 2008 GFC: -20.18% = 2.50x model VaR
- March 2020 COVID: -18.14% = 2.25x model VaR
- 2022 rate hikes: -10.34% = 1.28x model VaR

Even ES at 99% (11.33%) is breached in all three windows. The model holds in calm
regimes and breaks in crisis ones. Stressed ES under FRTB is the regulatory
response, though it still calibrates to a known crisis, not the next one.

## Setup

```bash
git clone git@github.com:t-x-hu/var-dashboard.git
cd var-dashboard
conda create -n var-dashboard python=3.12
conda activate var-dashboard
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

## Project structure

```
src/
  portfolio.py     positions, weights, $10mm notional
  returns.py       Yahoo Finance fetcher, daily and rolling returns
  var.py           Historical Simulation VaR/ES, rolling windows
  monte_carlo.py   Normal MC VaR/ES, configurable horizon
  stress.py        Crisis windows + per-window worst-loss / drawdown stats
  storage.py       SQLite persistence (positions, returns, risk run audit log)
notebooks/
  analysis.ipynb   Full narrative: portfolio, returns, VaR/ES, stress, conclusion


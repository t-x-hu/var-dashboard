# VaR / ES Dashboard

A small risk engine I built to compute Value at Risk and Expected Shortfall
on a 15-stock US equity portfolio. Two methods (historical simulation and
Monte Carlo), three crisis stress tests (2008, 2020 COVID, 2022 rate hikes).

Originally a project for my MAFN coursework at Columbia; rebuilding it
cleaner now as part of refreshing my portfolio.

## What's inside

- `src/` — the risk engine itself (portfolio, returns, VaR, ES, stress)
- `notebooks/` — one Jupyter notebook that pulls it all together for analysis
- `data/` — local SQLite cache (gitignored, regenerates on first run)

## Setup

Conda env, Python 3.12:

```bash
conda create -n var-dashboard python=3.12 -y
conda activate var-dashboard
pip install -r requirements.txt
```

## Notes

The 15 tickers were chosen to roughly mirror S&P 500 sector weights but
limited to large-cap names with clean daily data back through 2007 (so
the 2008 stress test has real history to draw from).

Historical Sim gave ~2.92% 1-day 99% VaR; Monte Carlo (normal assumption)
gave ~2.73%. The ~6% gap is the fat-tail premium — normal Monte Carlo
under-prices tail risk, which is exactly why ES under Basel III FRTB
exists.

Author: Tianxing Hu — tianxing.hu@outlook.com

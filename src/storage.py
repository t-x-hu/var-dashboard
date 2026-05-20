"""
SQLite persistence: positions, returns, risk_runs. Single file at
data/var_dashboard.db (gitignored).

Audit trail. Real risk systems log every run so breach analysis and
regulator questions can be answered later. Same logic here, smaller scale.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from portfolio import HOLDINGS, NOTIONAL

DB_PATH = Path("data/var_dashboard.db")


def init_db(path=DB_PATH):
    """Idempotent table creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            ticker          TEXT PRIMARY KEY,
            sector          TEXT NOT NULL,
            weight          REAL NOT NULL,
            dollar_position REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS returns (
            date             TEXT PRIMARY KEY,
            portfolio_return REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS risk_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp   TEXT NOT NULL,
            method          TEXT NOT NULL,
            horizon_days    INTEGER NOT NULL,
            confidence      REAL NOT NULL,
            var_pct         REAL NOT NULL,
            es_pct          REAL NOT NULL
        );
    """)
    conn.commit()
    return conn


def save_positions(conn):
    """Drop and rewrite. 15 rows so perf is moot."""
    rows = [(t, sec, w, NOTIONAL * w) for t, (sec, w) in HOLDINGS.items()]
    conn.execute("DELETE FROM positions")
    conn.executemany("INSERT INTO positions VALUES (?, ?, ?, ?)", rows)
    conn.commit()


def save_returns(conn, port_returns):
    """date is PK -> INSERT OR REPLACE keeps re-runs idempotent."""
    rows = [(d.strftime("%Y-%m-%d"), float(r)) for d, r in port_returns.items()]
    conn.executemany("INSERT OR REPLACE INTO returns VALUES (?, ?)", rows)
    conn.commit()


def save_risk_run(conn, method, horizon_days, confidence, var_pct, es_pct):
    # append-only log, one row per computation
    conn.execute(
        "INSERT INTO risk_runs (run_timestamp, method, horizon_days, confidence, var_pct, es_pct) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), method, horizon_days, confidence, var_pct, es_pct),
    )
    conn.commit()


def load_risk_history(conn):
    return conn.execute(
        "SELECT run_timestamp, method, horizon_days, confidence, var_pct, es_pct "
        "FROM risk_runs ORDER BY run_timestamp"
    ).fetchall()


if __name__ == "__main__":
    from returns import fetch_prices, daily_returns, portfolio_returns
    from var import historical_var, historical_es, rolling_returns
    from monte_carlo import mc_var, mc_es

    conn = init_db()
    save_positions(conn)

    print("fetching prices...")  # the only slow step
    prices = fetch_prices()
    rets = daily_returns(prices)
    port = portfolio_returns(rets)
    save_returns(conn, port)

    # log all four runs: hist & mc, 1d & 5d
    save_risk_run(conn, "historical",  1, 0.99, historical_var(port),    historical_es(port))
    save_risk_run(conn, "monte_carlo", 1, 0.99, mc_var(port),             mc_es(port))
    port_5d = rolling_returns(port, 5).dropna()
    save_risk_run(conn, "historical",  5, 0.99, historical_var(port_5d), historical_es(port_5d))
    save_risk_run(conn, "monte_carlo", 5, 0.99, mc_var(port, horizon=5), mc_es(port, horizon=5))

    print("\nrisk_runs in db:")
    for ts, method, h, conf, var, es in load_risk_history(conn):
        print(f"  {ts[:19]}  {method:<11}  {h}d  {conf:.0%}  VaR={var:.2%}  ES={es:.2%}")

    n_returns = conn.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
    n_positions = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    print(f"\nreturns:   {n_returns} rows")
    print(f"positions: {n_positions} rows")

    conn.close()
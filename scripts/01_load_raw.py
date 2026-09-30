"""
Load raw AML transactions into DuckDB with an explicit schema,
then run data-quality checks that stop the pipeline on failure.

Run from project root:  python scripts/01_load_raw.py
"""
from pathlib import Path
import sys
import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "HI-Small_Trans.csv"
DB_PATH = ROOT / "data" / "processed" / "aml.duckdb"
EXPECTED_ROWS = 5_078_345  # verified with `wc -l` minus header

# Explicit schema: fixes the duplicate "Account" header and keeps
# leading zeros in bank/account IDs by reading them as text.
# DECIMAL for money (DOUBLE introduces floating-point rounding errors).
# v2: DECIMAL(18,2) silently rounded Bitcoin amounts (e.g. 0.00123 BTC -> 0.00),
#     creating fake duplicates and hiding BTC from the FX analysis. Now 8 decimals.
SCHEMA = {
    "ts": "TIMESTAMP",
    "from_bank": "VARCHAR",
    "from_account": "VARCHAR",
    "to_bank": "VARCHAR",
    "to_account": "VARCHAR",
    "amount_received": "DECIMAL(24,8)",   # 8 decimals: Bitcoin amounts go to satoshi precision
    "receiving_currency": "VARCHAR",
    "amount_paid": "DECIMAL(24,8)",
    "payment_currency": "VARCHAR",
    "payment_format": "VARCHAR",
    "is_laundering": "TINYINT",
}


def load(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        CREATE OR REPLACE TABLE raw_transactions AS
        SELECT
            row_number() OVER () AS txn_id,   -- surrogate key: raw data has no ID
            *
        FROM read_csv(
            '{RAW_CSV}',
            header = true,
            columns = {SCHEMA},
            timestampformat = '%Y/%m/%d %H:%M'
        )
    """)


def q(con, sql):
    return con.execute(sql).fetchone()[0]


def run_checks(con) -> bool:
    """Hard checks: pipeline must stop if any fail."""
    checks = [
        ("Row count matches source",
         q(con, "SELECT count(*) FROM raw_transactions") == EXPECTED_ROWS),
        ("No NULLs in key columns",
         q(con, """SELECT count(*) FROM raw_transactions
                   WHERE ts IS NULL OR from_account IS NULL OR to_account IS NULL
                      OR amount_paid IS NULL OR is_laundering IS NULL""") == 0),
        ("No negative amounts",
         q(con, "SELECT count(*) FROM raw_transactions WHERE amount_paid < 0 OR amount_received < 0") == 0),
        ("Label is binary (0/1)",
         q(con, "SELECT count(*) FROM raw_transactions WHERE is_laundering NOT IN (0, 1)") == 0),
        ("txn_id is unique",
         q(con, "SELECT count(*) - count(DISTINCT txn_id) FROM raw_transactions") == 0),
    ]
    ok = True
    print("\n=== DATA QUALITY CHECKS ===")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    return ok


def profile(con) -> None:
    """Soft facts: not pass/fail, but they shape the analysis."""
    print("\n=== PROFILE ===")
    facts = {
        "Transactions": "SELECT count(*) FROM raw_transactions",
        "Laundering txns": "SELECT sum(is_laundering) FROM raw_transactions",
        "Laundering rate (%)": "SELECT round(100.0 * avg(is_laundering), 4) FROM raw_transactions",
        "Date range": "SELECT min(ts)::VARCHAR || ' to ' || max(ts)::VARCHAR FROM raw_transactions",
        "Distinct accounts": """SELECT count(DISTINCT acc) FROM (
                                  SELECT from_account AS acc FROM raw_transactions
                                  UNION SELECT to_account FROM raw_transactions)""",
        "Distinct banks": """SELECT count(DISTINCT b) FROM (
                               SELECT from_bank AS b FROM raw_transactions
                               UNION SELECT to_bank FROM raw_transactions)""",
        "Currencies": "SELECT count(DISTINCT payment_currency) FROM raw_transactions",
        "Cross-currency txns": "SELECT count(*) FROM raw_transactions WHERE payment_currency <> receiving_currency",
        "Self-transfers": "SELECT count(*) FROM raw_transactions WHERE from_account = to_account",
        "Exact duplicate rows": """SELECT count(*) - count(DISTINCT (ts, from_bank, from_account, to_bank,
                                   to_account, amount_paid, payment_currency, amount_received,
                                   receiving_currency, payment_format, is_laundering))
                                   FROM raw_transactions""",
    }
    for label, sql in facts.items():
        val = q(con, sql)
        print(f"  {label:<24} {val:,}" if isinstance(val, int) else f"  {label:<24} {val}")

    print("\n  Payment formats (count | laundering rate %):")
    for fmt, n, rate in con.execute("""
        SELECT payment_format, count(*), round(100.0 * avg(is_laundering), 3)
        FROM raw_transactions GROUP BY 1 ORDER BY 3 DESC
    """).fetchall():
        print(f"    {fmt:<14} {n:>10,} | {rate}")


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    print(f"Loading {RAW_CSV.name} into {DB_PATH.name} ...")
    load(con)
    ok = run_checks(con)
    profile(con)
    con.close()
    if not ok:
        sys.exit("\nData quality checks failed. Fix the issues above before continuing.")
    print("\nRaw load complete.")


if __name__ == "__main__":
    main()

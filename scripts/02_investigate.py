"""
Investigate data issues before cleaning.
Each cleaning decision (drop, flag or convert) is based on these results.

Run from project root:  python scripts/02_investigate.py
"""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "aml.duckdb"
con = duckdb.connect(str(DB_PATH), read_only=True)


def show(title: str, sql: str) -> None:
    print(f"\n=== {title} ===")
    con.sql(sql).show(max_rows=40)


# 1. Duplicates: are they junk, or do they carry laundering labels?
show("1. EXACT DUPLICATES by payment format & label", """
    WITH groups AS (
        SELECT ts, from_bank, from_account, to_bank, to_account,
               amount_paid, payment_currency, amount_received, receiving_currency,
               payment_format, is_laundering, count(*) AS n
        FROM raw_transactions
        GROUP BY ALL
        HAVING count(*) > 1
    )
    SELECT payment_format, is_laundering,
           count(*)     AS dup_groups,
           sum(n - 1)   AS extra_rows
    FROM groups
    GROUP BY ALL
    ORDER BY extra_rows DESC
""")

# 2. Self-transfers: which formats, and any laundering?
show("2. SELF-TRANSFERS (same bank + same account)", """
    SELECT payment_format,
           count(*)           AS self_txns,
           sum(is_laundering) AS laundering
    FROM raw_transactions
    WHERE from_bank = to_bank AND from_account = to_account
    GROUP BY 1
    ORDER BY 2 DESC
""")

show("2b. Same account ID but DIFFERENT bank (should be ~0)", """
    SELECT count(*) AS n
    FROM raw_transactions
    WHERE from_account = to_account AND from_bank <> to_bank
""")

show("2c. Accounts appearing under MORE THAN ONE bank anywhere (global key test)", """
    WITH acct_bank AS (
        SELECT from_account AS account, from_bank AS bank FROM raw_transactions
        UNION SELECT to_account, to_bank FROM raw_transactions
    )
    SELECT count(*) AS accounts_with_multiple_banks
    FROM (SELECT account FROM acct_bank GROUP BY 1 HAVING count(DISTINCT bank) > 1)
""")

# 3. Implied FX rates (USD per 1 unit of currency), derived from the data.
#    Filter on the USD leg (>= $10): cents-rounding makes tiny amounts noisy.
#    (v1 filtered the foreign leg, which silently excluded all Bitcoin: BTC amounts are < 10.)
show("3. IMPLIED FX RATES vs USD (derived from cross-currency txns)", """
    WITH obs AS (
        SELECT payment_currency AS currency,
               amount_received::DOUBLE / amount_paid::DOUBLE AS usd_per_unit
        FROM raw_transactions
        WHERE receiving_currency = 'US Dollar'
          AND payment_currency  <> 'US Dollar'
          AND amount_received >= 10 AND amount_paid > 0
        UNION ALL
        SELECT receiving_currency,
               amount_paid::DOUBLE / amount_received::DOUBLE
        FROM raw_transactions
        WHERE payment_currency   = 'US Dollar'
          AND receiving_currency <> 'US Dollar'
          AND amount_paid >= 10 AND amount_received > 0
    )
    SELECT currency,
           count(*)                        AS n_obs,
           round(median(usd_per_unit), 6)  AS median_rate,
           round(min(usd_per_unit), 6)     AS min_rate,
           round(max(usd_per_unit), 6)     AS max_rate
    FROM obs
    GROUP BY 1
    ORDER BY 1
""")

# 3b. Any currency with NO direct USD pair? (would need triangulation)
show("3b. Currencies WITHOUT a direct USD pair", """
    WITH all_ccy AS (
        SELECT payment_currency AS currency FROM raw_transactions
        UNION SELECT receiving_currency FROM raw_transactions
    ),
    usd_paired AS (
        SELECT payment_currency AS currency FROM raw_transactions
        WHERE receiving_currency = 'US Dollar'
        UNION SELECT receiving_currency FROM raw_transactions
        WHERE payment_currency = 'US Dollar'
    )
    SELECT a.currency
    FROM all_ccy a LEFT JOIN usd_paired u USING (currency)
    WHERE u.currency IS NULL AND a.currency <> 'US Dollar'
""")

# 4. Laundering by currency (volume view, not value yet)
show("4. LAUNDERING by payment currency", """
    SELECT payment_currency,
           count(*)                                AS txns,
           sum(is_laundering)                      AS laundering,
           round(100.0 * avg(is_laundering), 3)    AS laundering_rate_pct
    FROM raw_transactions
    GROUP BY 1
    ORDER BY laundering DESC
""")

con.close()

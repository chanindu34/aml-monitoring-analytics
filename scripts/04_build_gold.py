"""
Build the gold layer: a star schema for Tableau and the ML scripts.

Grain of fact_transactions: one row per unique transaction (duplicates excluded).

  fact_transactions
  dim_date, dim_account (role-playing: sender and receiver), dim_bank,
  dim_payment_format, dim_currency (role-playing: paid and received), dim_typology

Run from project root:  python scripts/04_build_gold.py
"""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "aml.duckdb"


# ------------------------------------------------------------ small dims
def build_dim_date(con):
    # Tail period = days whose volume is < 1% of the median daily volume.
    # Data-driven (not a hardcoded date): the simulation ends ~Sep 10, after which
    # only in-flight laundering chains complete, giving 57-73% laundering rates.
    con.execute("""
        CREATE OR REPLACE TABLE dim_date AS
        WITH daily AS (
            SELECT txn_date, count(*) AS txn_count
            FROM stg_transactions WHERE NOT is_duplicate
            GROUP BY 1
        )
        SELECT
            CAST(strftime(txn_date, '%Y%m%d') AS INTEGER) AS date_key,
            txn_date                                      AS date,
            dayname(txn_date)                             AS day_name,
            isodow(txn_date)                              AS day_of_week,   -- 1 = Mon
            isodow(txn_date) IN (6, 7)                    AS is_weekend,
            day(txn_date)                                 AS day_of_month,
            txn_count,
            txn_count < 0.01 * (SELECT median(txn_count) FROM daily) AS is_tail_period
        FROM daily
        ORDER BY date_key
    """)


def build_dim_payment_format(con):
    # channel_type is domain knowledge the data does not contain
    con.execute("""
        CREATE OR REPLACE TABLE dim_payment_format AS
        SELECT row_number() OVER (ORDER BY payment_format) AS payment_format_key, *
        FROM (VALUES
            ('ACH',          'Batch electronic'),
            ('Wire',         'Real-time electronic'),
            ('Cheque',       'Paper'),
            ('Credit Card',  'Card'),
            ('Cash',         'Cash'),
            ('Bitcoin',      'Crypto'),
            ('Reinvestment', 'Internal')
        ) AS t(payment_format, channel_type)
    """)


def build_dim_currency(con):
    con.execute("""
        CREATE OR REPLACE TABLE dim_currency AS
        WITH iso AS (
            SELECT * FROM (VALUES
                ('US Dollar', 'USD'), ('Euro', 'EUR'), ('Yuan', 'CNY'), ('Yen', 'JPY'),
                ('UK Pound', 'GBP'), ('Swiss Franc', 'CHF'), ('Rupee', 'INR'),
                ('Ruble', 'RUB'), ('Canadian Dollar', 'CAD'), ('Australian Dollar', 'AUD'),
                ('Mexican Peso', 'MXN'), ('Brazil Real', 'BRL'), ('Saudi Riyal', 'SAR'),
                ('Shekel', 'ILS'), ('Bitcoin', 'BTC')
            ) AS t(currency, iso_code)
        )
        SELECT row_number() OVER (ORDER BY f.currency) AS currency_key,
               f.currency, i.iso_code, f.usd_per_unit AS usd_rate, f.n_obs AS fx_observations
        FROM stg_fx_rates f
        LEFT JOIN iso i USING (currency)
    """)


def build_dim_typology(con):
    # Hand-authored AML domain knowledge. detection_difficulty is a hypothesis,
    # tested against rule performance in 07_rule_simulation.py.
    con.execute("""
        CREATE OR REPLACE TABLE dim_typology AS
        SELECT row_number() OVER () - 1 AS typology_key, *
        FROM (VALUES
            ('NONE',           'Legitimate transaction (not laundering)',
                               'N/A',        'N/A'),
            ('FAN-OUT',        'One account splits funds to many accounts',
                               'Layering',   'Medium'),
            ('FAN-IN',         'Many accounts send funds into one account',
                               'Placement',  'Medium'),
            ('GATHER-SCATTER', 'Funds collected into a hub, then redistributed',
                               'Layering',   'Medium'),
            ('SCATTER-GATHER', 'Funds split across intermediaries, then reconverge',
                               'Layering',   'Hard'),
            ('CYCLE',          'Funds hop through a chain and return to origin',
                               'Layering',   'Hard'),
            ('BIPARTITE',      'A group of senders pays a group of receivers',
                               'Layering',   'Hard'),
            ('STACK',          'Two bipartite layers chained together',
                               'Layering',   'Hard'),
            ('RANDOM',         'Funds follow a random walk through accounts',
                               'Layering',   'Very hard'),
            ('UNDOCUMENTED',   'Labelled laundering with no documented typology',
                               'Unknown',    'Very hard')
        ) AS t(typology, description, aml_stage, detection_difficulty)
    """)


# ------------------------------------------------------------ big dims
def build_dim_account(con):
    con.execute("""
        CREATE OR REPLACE TABLE dim_account AS
        WITH txns AS (SELECT * FROM stg_transactions WHERE NOT is_duplicate),
        outgoing AS (
            SELECT from_account_key AS account_key,
                   any_value(from_bank)    AS bank_id,
                   any_value(from_account) AS account_id,
                   count(*)                AS txns_out,
                   count(DISTINCT to_account_key) FILTER (WHERE NOT is_self_transfer)
                                           AS counterparties_out,
                   sum(amount_paid_usd)    AS usd_out,
                   min(ts) AS first_out, max(ts) AS last_out,
                   sum(is_laundering)      AS label_laundering_txns_out
            FROM txns GROUP BY 1
        ),
        incoming AS (
            SELECT to_account_key AS account_key,
                   any_value(to_bank)      AS bank_id,
                   any_value(to_account)   AS account_id,
                   count(*)                AS txns_in,
                   count(DISTINCT from_account_key) FILTER (WHERE NOT is_self_transfer)
                                           AS counterparties_in,
                   sum(amount_received_usd) AS usd_in,
                   min(ts) AS first_in, max(ts) AS last_in,
                   sum(is_laundering)      AS label_laundering_txns_in
            FROM txns GROUP BY 1
        )
        SELECT
            row_number() OVER (ORDER BY account_key)          AS account_sk,
            account_key,
            coalesce(o.bank_id, i.bank_id)                    AS bank_id,
            coalesce(o.account_id, i.account_id)              AS account_id,
            coalesce(o.txns_out, 0)                           AS txns_out,
            coalesce(i.txns_in, 0)                            AS txns_in,
            coalesce(o.counterparties_out, 0)                 AS counterparties_out,
            coalesce(i.counterparties_in, 0)                  AS counterparties_in,
            coalesce(o.usd_out, 0)                            AS usd_out,
            coalesce(i.usd_in, 0)                             AS usd_in,
            least(o.first_out, i.first_in)                    AS first_seen,
            greatest(o.last_out, i.last_in)                   AS last_seen,
            -- label-derived: for dashboards only, never used as ML features
            coalesce(o.label_laundering_txns_out, 0)          AS label_laundering_txns_out,
            coalesce(i.label_laundering_txns_in, 0)           AS label_laundering_txns_in
        FROM outgoing o
        FULL OUTER JOIN incoming i USING (account_key)
    """)


def build_dim_bank(con):
    con.execute("""
        CREATE OR REPLACE TABLE dim_bank AS
        SELECT bank_id, count(*) AS n_accounts
        FROM dim_account
        GROUP BY 1
    """)


# ------------------------------------------------------------ fact
def build_fact(con):
    con.execute("""
        CREATE OR REPLACE TABLE fact_transactions AS
        SELECT
            t.txn_id,
            CAST(strftime(t.txn_date, '%Y%m%d') AS INTEGER) AS date_key,
            t.ts,
            t.txn_hour,
            fa.account_sk          AS from_account_sk,
            ta.account_sk          AS to_account_sk,
            t.from_bank            AS from_bank_id,
            t.to_bank              AS to_bank_id,
            pf.payment_format_key,
            pc.currency_key        AS payment_currency_key,
            rc.currency_key        AS receiving_currency_key,
            ty.typology_key,
            t.attempt_id,
            t.amount_paid          AS amount_paid_orig,
            t.amount_paid_usd,
            t.amount_received_usd,
            t.is_laundering,
            t.is_self_transfer,
            t.is_cross_currency
        FROM stg_transactions t
        LEFT JOIN dim_account fa        ON t.from_account_key   = fa.account_key
        LEFT JOIN dim_account ta        ON t.to_account_key     = ta.account_key
        LEFT JOIN dim_payment_format pf ON t.payment_format     = pf.payment_format
        LEFT JOIN dim_currency pc       ON t.payment_currency   = pc.currency
        LEFT JOIN dim_currency rc       ON t.receiving_currency = rc.currency
        LEFT JOIN dim_typology ty       ON ty.typology = coalesce(
                                              t.typology,
                                              CASE WHEN t.is_laundering = 1
                                                   THEN 'UNDOCUMENTED' ELSE 'NONE' END)
        WHERE NOT t.is_duplicate
        ORDER BY t.txn_id
    """)


# ------------------------------------------------------------ checks
def q(con, sql):
    return con.execute(sql).fetchone()[0]


def run_checks(con) -> bool:
    silver = "FROM stg_transactions WHERE NOT is_duplicate"
    checks = [
        ("Grain: fact rows = unique silver rows",
         q(con, "SELECT count(*) FROM fact_transactions") == q(con, f"SELECT count(*) {silver}")),
        ("Reconciliation: USD total matches silver",
         q(con, "SELECT sum(amount_paid_usd) FROM fact_transactions")
         == q(con, f"SELECT sum(amount_paid_usd) {silver}")),
        ("Reconciliation: laundering count matches silver",
         q(con, "SELECT sum(is_laundering) FROM fact_transactions")
         == q(con, f"SELECT sum(is_laundering) {silver}")),
        ("Referential integrity: no NULL foreign keys",
         q(con, """SELECT count(*) FROM fact_transactions
                   WHERE date_key IS NULL OR from_account_sk IS NULL OR to_account_sk IS NULL
                      OR payment_format_key IS NULL OR payment_currency_key IS NULL
                      OR receiving_currency_key IS NULL OR typology_key IS NULL""") == 0),
        ("Every date_key exists in dim_date",
         q(con, """SELECT count(*) FROM fact_transactions f
                   ANTI JOIN dim_date d USING (date_key)""") == 0),
        ("dim_account: account_key is unique",
         q(con, "SELECT count(*) - count(DISTINCT account_key) FROM dim_account") == 0),
        ("dim_currency: every currency has an ISO code",
         q(con, "SELECT count(*) FROM dim_currency WHERE iso_code IS NULL") == 0),
        ("Typed + undocumented laundering = all laundering",
         q(con, """SELECT count(*) FROM fact_transactions f JOIN dim_typology t USING (typology_key)
                   WHERE t.typology <> 'NONE'""")
         == q(con, "SELECT sum(is_laundering) FROM fact_transactions")),
    ]
    ok = True
    print("\n=== DATA QUALITY CHECKS ===")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    return ok


def profile(con):
    print("\n=== GOLD TABLES ===")
    for t in ["fact_transactions", "dim_account", "dim_bank", "dim_date",
              "dim_payment_format", "dim_currency", "dim_typology"]:
        print(f"  {t:<22} {q(con, f'SELECT count(*) FROM {t}'):>10,} rows")

    print("\n=== STAR SCHEMA CHECK: laundering by typology and AML stage ===")
    con.sql("""
        SELECT t.typology, t.aml_stage, t.detection_difficulty,
               count(*)                                   AS txns,
               round(median(f.amount_paid_usd), 0)        AS median_usd,
               round(sum(f.amount_paid_usd) / 1e6, 1)     AS total_usd_m
        FROM fact_transactions f
        JOIN dim_typology t USING (typology_key)
        WHERE t.typology <> 'NONE'
        GROUP BY ALL
        ORDER BY txns DESC
    """).show()


def main():
    con = duckdb.connect(str(DB_PATH))
    print("Building dimensions ...")
    build_dim_date(con)
    build_dim_payment_format(con)
    build_dim_currency(con)
    build_dim_typology(con)
    build_dim_account(con)
    build_dim_bank(con)
    print("Building fact_transactions ...")
    build_fact(con)
    ok = run_checks(con)
    profile(con)
    con.close()
    if not ok:
        sys.exit("\nData quality checks failed. Fix the issues above before continuing.")
    print("Gold layer (star schema) complete.")


if __name__ == "__main__":
    main()

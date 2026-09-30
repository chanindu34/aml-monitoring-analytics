"""
Build the silver (staging) layer.

  stg_fx_rates      USD rate per currency, derived from the data's own cross-currency txns
  stg_patterns      parsed laundering attempts (attempt_id + typology per transaction)
  stg_transactions  cleaned txns + USD amounts + composite keys + flags + typology

Run from project root:  python scripts/03_build_staging.py
"""
import csv
import re
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "aml.duckdb"
PATTERNS_TXT = ROOT / "data" / "raw" / "HI-Small_Patterns.txt"
PATTERNS_CSV = ROOT / "data" / "processed" / "patterns_parsed.csv"

# Same explicit schema as the raw load.
TXN_SCHEMA = {
    "ts": "TIMESTAMP",
    "from_bank": "VARCHAR",
    "from_account": "VARCHAR",
    "to_bank": "VARCHAR",
    "to_account": "VARCHAR",
    "amount_received": "DECIMAL(24,8)",
    "receiving_currency": "VARCHAR",
    "amount_paid": "DECIMAL(24,8)",
    "payment_currency": "VARCHAR",
    "payment_format": "VARCHAR",
    "is_laundering": "TINYINT",
}
PATTERN_SCHEMA = {"attempt_id": "INTEGER", "typology": "VARCHAR",
                  "typology_detail": "VARCHAR", **TXN_SCHEMA}

# e.g. "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out"
#      "BEGIN LAUNDERING ATTEMPT - STACK"
BEGIN_RE = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)(?::\s*(.*))?$")

# Every column that defines "the same transaction" (everything except txn_id)
MATCH_KEYS = ["ts", "from_bank", "from_account", "to_bank", "to_account",
              "amount_paid", "payment_currency", "amount_received",
              "receiving_currency", "payment_format"]


# ---------------------------------------------------------------- patterns
def parse_patterns() -> int:
    """Text file with BEGIN/END blocks -> flat CSV with attempt_id + typology."""
    rows, attempt_id, typology, detail = [], 0, None, None
    with open(PATTERNS_TXT) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            m = BEGIN_RE.match(line)
            if m:
                attempt_id += 1
                typology, detail = m.group(1), (m.group(2) or "").strip()
            elif line.startswith("END LAUNDERING ATTEMPT"):
                typology = None
            elif typology:
                fields = line.split(",")
                if len(fields) != 11:
                    sys.exit(f"Patterns line {line_no}: expected 11 fields, got {len(fields)}")
                rows.append([attempt_id, typology, detail] + fields)
            else:
                sys.exit(f"Patterns line {line_no}: data row outside a BEGIN/END block")

    with open(PATTERNS_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return attempt_id


def build_patterns(con) -> None:
    con.execute(f"""
        CREATE OR REPLACE TABLE stg_patterns AS
        SELECT * FROM read_csv(
            '{PATTERNS_CSV}',
            header = false,
            columns = {PATTERN_SCHEMA},
            timestampformat = '%Y/%m/%d %H:%M'
        )
    """)


# ---------------------------------------------------------------- FX rates
def build_fx(con) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE stg_fx_rates AS
        WITH obs AS (
            SELECT payment_currency AS currency,
                   amount_received::DOUBLE / amount_paid::DOUBLE AS usd_per_unit
            FROM raw_transactions
            WHERE receiving_currency = 'US Dollar' AND payment_currency <> 'US Dollar'
              AND amount_received >= 10 AND amount_paid > 0
            UNION ALL
            SELECT receiving_currency,
                   amount_paid::DOUBLE / amount_received::DOUBLE
            FROM raw_transactions
            WHERE payment_currency = 'US Dollar' AND receiving_currency <> 'US Dollar'
              AND amount_paid >= 10 AND amount_received > 0
        )
        SELECT currency, median(usd_per_unit) AS usd_per_unit, count(*) AS n_obs
        FROM obs GROUP BY 1
        UNION ALL
        SELECT 'US Dollar', 1.0, NULL
    """)


# ---------------------------------------------------------------- transactions
def build_transactions(con) -> None:
    partition = ", ".join(f"r.{k}" for k in MATCH_KEYS + ["is_laundering"])
    join_on = " AND ".join(f"b.{k} = p.{k}" for k in MATCH_KEYS)

    con.execute(f"""
        CREATE OR REPLACE TABLE stg_transactions AS
        WITH base AS (
            SELECT r.*,
                   r.from_bank || '_' || r.from_account AS from_account_key,
                   r.to_bank   || '_' || r.to_account   AS to_account_key,
                   -- 1st copy = original, 2nd+ copies = duplicates
                   row_number() OVER (PARTITION BY {partition} ORDER BY r.txn_id) > 1
                       AS is_duplicate
            FROM raw_transactions r
        ),
        typ AS (
            SELECT b.txn_id, p.attempt_id, p.typology
            FROM base b
            JOIN stg_patterns p ON {join_on}
            -- if a txn appears in several attempts, keep the first
            QUALIFY row_number() OVER (PARTITION BY b.txn_id ORDER BY p.attempt_id) = 1
        )
        SELECT
            b.txn_id,
            b.ts,
            CAST(b.ts AS DATE)  AS txn_date,
            hour(b.ts)          AS txn_hour,
            dayname(b.ts)       AS txn_weekday,
            b.from_bank, b.from_account, b.from_account_key,
            b.to_bank,   b.to_account,   b.to_account_key,
            b.amount_paid, b.payment_currency,
            CAST(round(b.amount_paid::DOUBLE * fp.usd_per_unit, 2) AS DECIMAL(24,2))
                AS amount_paid_usd,
            b.amount_received, b.receiving_currency,
            CAST(round(b.amount_received::DOUBLE * fr.usd_per_unit, 2) AS DECIMAL(24,2))
                AS amount_received_usd,
            b.payment_format,
            b.is_laundering,
            (b.from_account_key = b.to_account_key)        AS is_self_transfer,
            (b.payment_currency <> b.receiving_currency)   AS is_cross_currency,
            b.is_duplicate,
            t.attempt_id,
            t.typology
        FROM base b
        LEFT JOIN stg_fx_rates fp ON b.payment_currency   = fp.currency
        LEFT JOIN stg_fx_rates fr ON b.receiving_currency = fr.currency
        LEFT JOIN typ t USING (txn_id)
        ORDER BY b.txn_id
    """)


# ---------------------------------------------------------------- checks
def q(con, sql):
    return con.execute(sql).fetchone()[0]


def run_checks(con) -> bool:
    raw_rows = q(con, "SELECT count(*) FROM raw_transactions")
    expected_dups = q(con, f"""
        SELECT count(*) - count(DISTINCT ({', '.join(MATCH_KEYS)}, is_laundering))
        FROM raw_transactions""")
    checks = [
        ("Row count preserved (raw = staging)",
         q(con, "SELECT count(*) FROM stg_transactions") == raw_rows),
        ("Every currency has an FX rate",
         q(con, """SELECT count(*) FROM stg_transactions
                   WHERE amount_paid_usd IS NULL OR amount_received_usd IS NULL""") == 0),
        ("FX table covers all 15 currencies",
         q(con, "SELECT count(*) FROM stg_fx_rates") == 15),
        ("Duplicate flags match raw duplicate count",
         q(con, "SELECT count(*) FROM stg_transactions WHERE is_duplicate") == expected_dups),
        ("Patterns parsed into rows",
         q(con, "SELECT count(*) FROM stg_patterns") > 0),
    ]
    ok = True
    print("\n=== DATA QUALITY CHECKS ===")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    return ok


def profile(con, n_attempts: int) -> None:
    join_on = " AND ".join(f"r.{k} = p.{k}" for k in MATCH_KEYS)
    pattern_rows = q(con, "SELECT count(*) FROM stg_patterns")
    matched = q(con, f"""
        SELECT count(*) FROM stg_patterns p
        WHERE EXISTS (SELECT 1 FROM raw_transactions r WHERE {join_on})""")

    print("\n=== PATTERNS TO TRANSACTIONS LINKAGE ===")
    print(f"  Laundering attempts parsed      {n_attempts:,}")
    print(f"  Pattern rows                    {pattern_rows:,}")
    print(f"  Pattern rows found in raw       {matched:,}  ({100 * matched / pattern_rows:.1f}%)")
    print(f"  Laundering txns WITH typology   "
          f"{q(con, 'SELECT count(*) FROM stg_transactions WHERE is_laundering = 1 AND typology IS NOT NULL'):,}")
    print(f"  Laundering txns WITHOUT typology "
          f"{q(con, 'SELECT count(*) FROM stg_transactions WHERE is_laundering = 1 AND typology IS NULL'):,}")
    print(f"  Typology-tagged but label = 0   "
          f"{q(con, 'SELECT count(*) FROM stg_transactions WHERE typology IS NOT NULL AND is_laundering = 0'):,}")

    print("\n=== TYPOLOGY SUMMARY ===")
    con.sql("""
        SELECT typology,
               count(DISTINCT attempt_id)          AS attempts,
               count(*)                            AS txns,
               round(sum(amount_paid_usd) / 1e6, 2) AS usd_millions
        FROM stg_transactions
        WHERE typology IS NOT NULL
        GROUP BY 1 ORDER BY txns DESC
    """).show()

    print("=== MONEY VIEW (USD, excluding duplicates) ===")
    con.sql("""
        SELECT round(sum(amount_paid_usd) / 1e9, 2)                               AS total_usd_bn,
               round(sum(amount_paid_usd) FILTER (WHERE is_laundering = 1) / 1e6, 2) AS laundering_usd_m,
               round(100.0 * sum(amount_paid_usd) FILTER (WHERE is_laundering = 1)
                     / sum(amount_paid_usd), 4)                                     AS laundering_value_pct
        FROM stg_transactions
        WHERE NOT is_duplicate
    """).show()


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    print("Parsing patterns file ...")
    n_attempts = parse_patterns()
    print("Building stg_patterns, stg_fx_rates, stg_transactions ...")
    build_patterns(con)
    build_fx(con)
    build_transactions(con)
    ok = run_checks(con)
    profile(con, n_attempts)
    con.close()
    if not ok:
        sys.exit("\nData quality checks failed. Fix the issues above before continuing.")
    print("Silver layer complete.")


if __name__ == "__main__":
    main()

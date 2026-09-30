"""
Risk segmentation: where is laundering concentrated?

For every segment: laundering rate, 95% Wilson CI, lift vs base rate,
two-proportion z-test vs the rest, Bonferroni-corrected significance.
The statistics are implemented directly (no scipy dependency).

Run from project root:  python scripts/05_analysis_risk.py
Outputs:                outputs/analysis/*.csv
"""
import csv
from math import erfc, sqrt
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "aml.duckdb"
OUT_DIR = ROOT / "outputs" / "analysis"
Z = 1.96  # 95% confidence


# ------------------------------------------------------------ statistics
def wilson(x: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Stable for rare events."""
    if n == 0:
        return 0.0, 0.0
    p = x / n
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = Z * sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def two_prop_pvalue(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided two-proportion z-test: is group 1's rate different from group 2's?"""
    if n1 == 0 or n2 == 0:
        return 1.0
    pooled = (x1 + x2) / (n1 + n2)
    se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return erfc(abs(z) / sqrt(2))


# ------------------------------------------------------------ segment analysis
def rate_table(con, title: str, group_expr: str, filename: str, sort_by_lift: bool = True):
    rows = con.execute(f"""
        SELECT {group_expr} AS grp, count(*) AS n, sum(is_laundering)::BIGINT AS x
        FROM v_txn GROUP BY 1
    """).fetchall()

    N = sum(r[1] for r in rows)
    X = sum(r[2] for r in rows)
    base = X / N
    alpha = 0.05 / len(rows)  # Bonferroni

    results = []
    for grp, n, x in rows:
        lo, hi = wilson(x, n)
        p_val = two_prop_pvalue(x, n, X - x, N - n)
        results.append({
            "group": grp, "txns": n, "laundering": x,
            "rate_pct": 100 * x / n, "ci_low_pct": 100 * lo, "ci_high_pct": 100 * hi,
            "lift": (x / n) / base, "p_value": p_val, "significant": p_val < alpha,
        })
    results.sort(key=(lambda r: -r["lift"]) if sort_by_lift else (lambda r: r["group"]))

    print(f"\n=== {title} ===")
    print(f"  base rate {100 * base:.4f}%  |  Bonferroni alpha = 0.05/{len(rows)} = {alpha:.5f}"
          f"  |  * = significant")
    print(f"  {'group':<20}{'txns':>11}{'laund':>7}{'rate%':>9}   {'95% CI %':<17}{'lift':>7}{'p-value':>11}")
    for r in results:
        ci = f"[{r['ci_low_pct']:.3f}-{r['ci_high_pct']:.3f}]"
        print(f"  {str(r['group']):<20}{r['txns']:>11,}{r['laundering']:>7,}{r['rate_pct']:>9.3f}"
              f"   {ci:<17}{r['lift']:>6.2f}x{r['p_value']:>11.1e} {'*' if r['significant'] else ''}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{filename}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)


def amount_quantiles(con):
    print("\n=== AMOUNT QUANTILES (USD): laundering vs legitimate ===")
    con.sql("""
        SELECT CASE WHEN is_laundering = 1 THEN 'Laundering' ELSE 'Legitimate' END AS label,
               round(quantile_cont(amount_paid_usd::DOUBLE, 0.10), 0) AS p10,
               round(quantile_cont(amount_paid_usd::DOUBLE, 0.25), 0) AS p25,
               round(quantile_cont(amount_paid_usd::DOUBLE, 0.50), 0) AS median,
               round(quantile_cont(amount_paid_usd::DOUBLE, 0.75), 0) AS p75,
               round(quantile_cont(amount_paid_usd::DOUBLE, 0.90), 0) AS p90,
               round(quantile_cont(amount_paid_usd::DOUBLE, 0.99), 0) AS p99,
               round(max(amount_paid_usd), 0)                          AS max
        FROM v_txn GROUP BY 1 ORDER BY 1
    """).show()


def main():
    con = duckdb.connect(str(DB_PATH))

    print("=== TAIL PERIOD (excluded from all rates below) ===")
    con.sql("""
        SELECT is_tail_period,
               min(d.date) AS first_day, max(d.date) AS last_day,
               count(*) AS txns, sum(is_laundering) AS laundering,
               round(100.0 * avg(is_laundering), 3) AS rate_pct
        FROM fact_transactions f JOIN dim_date d USING (date_key)
        GROUP BY 1 ORDER BY 1
    """).show()

    # Analysis window = non-tail days only
    con.execute("""
        CREATE OR REPLACE TEMP VIEW v_txn AS
        SELECT f.*, pf.payment_format, pf.channel_type,
               pc.currency AS payment_currency, d.date, d.day_name, d.is_weekend
        FROM fact_transactions f
        JOIN dim_payment_format pf USING (payment_format_key)
        JOIN dim_currency pc ON f.payment_currency_key = pc.currency_key
        JOIN dim_date d USING (date_key)
        WHERE NOT d.is_tail_period
    """)

    rate_table(con, "1. PAYMENT FORMAT", "payment_format", "risk_by_payment_format")
    rate_table(con, "2. PAYMENT CURRENCY", "payment_currency", "risk_by_currency")
    rate_table(con, "3. HOUR OF DAY", "txn_hour", "risk_by_hour", sort_by_lift=False)
    rate_table(con, "4. DATE", "date::VARCHAR", "risk_by_date", sort_by_lift=False)
    rate_table(con, "4b. WEEKEND vs WEEKDAY",
               "CASE WHEN is_weekend THEN 'Weekend' ELSE 'Weekday' END", "risk_by_weekend")
    rate_table(con, "5. CROSS-CURRENCY", "CASE WHEN is_cross_currency THEN 'Cross' ELSE 'Same' END",
               "risk_by_cross_currency")
    rate_table(con, "6. AMOUNT BUCKET (USD)", """
        CASE WHEN amount_paid_usd <     100 THEN '1. < $100'
             WHEN amount_paid_usd <   1000  THEN '2. $100-1K'
             WHEN amount_paid_usd <   5000  THEN '3. $1K-5K'
             WHEN amount_paid_usd <   9000  THEN '4. $5K-9K'
             WHEN amount_paid_usd <  10000  THEN '5. $9K-10K'
             WHEN amount_paid_usd <  11000  THEN '6. $10K-11K'
             WHEN amount_paid_usd <  50000  THEN '7. $11K-50K'
             WHEN amount_paid_usd < 1000000 THEN '8. $50K-1M'
             ELSE                                '9. > $1M' END""",
               "risk_by_amount_bucket", sort_by_lift=False)
    amount_quantiles(con)
    con.close()
    print(f"\nTables saved to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

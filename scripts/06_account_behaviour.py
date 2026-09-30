"""
Account-level behaviour: how do laundering accounts differ from normal ones?

Builds feat_account (one row per account, analysis window only), then:
  1. compares laundering vs normal accounts (medians + univariate AUC)
  2. validates features against known typologies (fingerprint sanity check)

Run from project root:  python scripts/06_account_behaviour.py
Outputs:                feat_account table in aml.duckdb, outputs/analysis/*.csv
"""
import csv
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "aml.duckdb"
OUT_DIR = ROOT / "outputs" / "analysis"

FEATURES = [
    "txns_out", "txns_in", "out_degree", "in_degree",
    "usd_out", "usd_in", "pass_through_ratio", "median_amt_out",
    "ach_share_out", "n_formats_out", "midband_share_out", "weekend_share_out",
    "max_txns_out_per_hour", "hours_in_to_out", "active_span_hours",
]


def build_features(con):
    con.execute("""
        CREATE OR REPLACE TABLE feat_account AS
        WITH f AS (
            SELECT f.*, pf.payment_format, d.is_weekend
            FROM fact_transactions f
            JOIN dim_date d            USING (date_key)
            JOIN dim_payment_format pf USING (payment_format_key)
            WHERE NOT d.is_tail_period
        ),
        outg AS (
            SELECT from_account_sk AS account_sk,
                   count(*)                                                        AS txns_out,
                   count(DISTINCT to_account_sk) FILTER (WHERE NOT is_self_transfer) AS out_degree,
                   sum(amount_paid_usd)::DOUBLE                                    AS usd_out,
                   median(amount_paid_usd)::DOUBLE                                 AS median_amt_out,
                   avg((payment_format = 'ACH')::INT)                              AS ach_share_out,
                   count(DISTINCT payment_format)                                  AS n_formats_out,
                   avg((amount_paid_usd BETWEEN 1000 AND 50000)::INT)              AS midband_share_out,
                   avg(is_weekend::INT)                                            AS weekend_share_out,
                   min(ts) AS first_out, max(ts) AS last_out,
                   max(is_laundering)                                              AS lbl_out
            FROM f GROUP BY 1
        ),
        inc AS (
            SELECT to_account_sk AS account_sk,
                   count(*)                                                          AS txns_in,
                   count(DISTINCT from_account_sk) FILTER (WHERE NOT is_self_transfer) AS in_degree,
                   sum(amount_received_usd)::DOUBLE                                  AS usd_in,
                   min(ts) AS first_in, max(ts) AS last_in,
                   max(is_laundering)                                                AS lbl_in
            FROM f GROUP BY 1
        ),
        base AS (
            SELECT coalesce(o.account_sk, i.account_sk) AS account_sk,
                   o.* EXCLUDE (account_sk),
                   i.* EXCLUDE (account_sk)
            FROM outg o FULL OUTER JOIN inc i ON o.account_sk = i.account_sk
        ),
        burst AS (
            SELECT from_account_sk AS account_sk, max(n) AS max_txns_out_per_hour
            FROM (SELECT from_account_sk, date_trunc('hour', ts) AS h, count(*) AS n
                  FROM f GROUP BY 1, 2)
            GROUP BY 1
        ),
        dwell AS (   -- hours from first money IN to first money OUT afterwards
            SELECT i.account_sk,
                   date_diff('minute', i.first_in, min(o.ts)) / 60.0 AS hours_in_to_out
            FROM inc i
            JOIN f o ON o.from_account_sk = i.account_sk
                    AND o.ts >= i.first_in
                    AND NOT o.is_self_transfer
            GROUP BY i.account_sk, i.first_in
        )
        SELECT
            b.account_sk,
            coalesce(txns_out, 0)    AS txns_out,
            coalesce(txns_in, 0)     AS txns_in,
            coalesce(out_degree, 0)  AS out_degree,
            coalesce(in_degree, 0)   AS in_degree,
            coalesce(usd_out, 0)     AS usd_out,
            coalesce(usd_in, 0)      AS usd_in,
            CASE WHEN greatest(coalesce(usd_in, 0), coalesce(usd_out, 0)) > 0
                 THEN least(coalesce(usd_in, 0), coalesce(usd_out, 0))
                    / greatest(coalesce(usd_in, 0), coalesce(usd_out, 0))
                 ELSE 0 END          AS pass_through_ratio,
            median_amt_out,
            ach_share_out,
            n_formats_out,
            midband_share_out,
            weekend_share_out,
            coalesce(bu.max_txns_out_per_hour, 0) AS max_txns_out_per_hour,
            dw.hours_in_to_out,
            date_diff('minute',
                      least(coalesce(first_out, first_in), coalesce(first_in, first_out)),
                      greatest(coalesce(last_out, last_in), coalesce(last_in, last_out))
                     ) / 60.0        AS active_span_hours,
            -- Target (label-derived): never used as a feature
            greatest(coalesce(lbl_out, 0), coalesce(lbl_in, 0)) AS label_laundering_account
        FROM base b
        LEFT JOIN burst bu USING (account_sk)
        LEFT JOIN dwell dw USING (account_sk)
    """)


def q(con, sql):
    return con.execute(sql).fetchone()[0]


def compare_features(con):
    n_acc = q(con, "SELECT count(*) FROM feat_account")
    n_lau = q(con, "SELECT sum(label_laundering_account) FROM feat_account")
    print(f"\nAccounts in window: {n_acc:,}  |  laundering-involved: {n_lau:,} "
          f"({100 * n_lau / n_acc:.3f}%)")

    results = []
    for feat in FEATURES:
        med0, med1 = con.execute(f"""
            SELECT median({feat}) FILTER (WHERE label_laundering_account = 0),
                   median({feat}) FILTER (WHERE label_laundering_account = 1)
            FROM feat_account WHERE {feat} IS NOT NULL
        """).fetchone()
        # AUC via average ranks (Mann-Whitney U). Ties get the average rank.
        auc = q(con, f"""
            WITH r AS (
                SELECT label_laundering_account AS y,
                       rank() OVER (ORDER BY {feat})
                       + (count(*) OVER (PARTITION BY {feat}) - 1) / 2.0 AS rk
                FROM feat_account WHERE {feat} IS NOT NULL
            ),
            c AS (SELECT count(*) FILTER (WHERE y = 1)::DOUBLE AS n1,
                         count(*) FILTER (WHERE y = 0)::DOUBLE AS n0 FROM r)
            SELECT ((SELECT sum(rk) FROM r WHERE y = 1) - n1 * (n1 + 1) / 2) / (n1 * n0)
            FROM c
        """)
        results.append({
            "feature": feat,
            "median_normal": med0, "median_laundering": med1,
            "auc": auc,
            "direction": "higher = riskier" if auc >= 0.5 else "lower = riskier",
            "strength": abs(auc - 0.5) * 2,   # 0 = useless, 1 = perfect separation
        })
    results.sort(key=lambda r: -r["strength"])

    print("\n=== LAUNDERING vs NORMAL ACCOUNTS (sorted by signal strength) ===")
    print(f"  {'feature':<24}{'median normal':>15}{'median laund.':>15}{'AUC':>8}   direction")
    for r in results:
        fmt = lambda v: "n/a" if v is None else f"{v:,.2f}"
        print(f"  {r['feature']:<24}{fmt(r['median_normal']):>15}{fmt(r['median_laundering']):>15}"
              f"{r['auc']:>8.3f}   {r['direction']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "account_feature_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)


def typology_fingerprint(con):
    print("\n=== TYPOLOGY FINGERPRINT (features checked against known patterns) ===")
    rel = con.sql("""
        SELECT t.typology,
               count(*)                          AS txns,
               median(s.out_degree)              AS sender_out_degree,
               median(r.in_degree)               AS receiver_in_degree,
               round(median(s.pass_through_ratio), 2) AS sender_pass_through,
               round(median(r.pass_through_ratio), 2) AS receiver_pass_through,
               round(median(s.hours_in_to_out), 1)    AS sender_dwell_hours
        FROM fact_transactions f
        JOIN dim_date d      USING (date_key)
        JOIN dim_typology t  USING (typology_key)
        JOIN feat_account s  ON f.from_account_sk = s.account_sk
        JOIN feat_account r  ON f.to_account_sk   = r.account_sk
        WHERE NOT d.is_tail_period
        GROUP BY 1
        ORDER BY (t.typology = 'NONE') DESC, txns DESC
    """)
    rel.show(max_rows=20)
    rel.write_csv(str(OUT_DIR / "typology_fingerprint.csv"))


def main():
    con = duckdb.connect(str(DB_PATH))
    print("Building feat_account ...")
    build_features(con)
    compare_features(con)
    typology_fingerprint(con)
    con.close()
    print(f"\nfeat_account built; tables saved to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

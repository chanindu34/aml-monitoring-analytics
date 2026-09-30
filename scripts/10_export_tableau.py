"""
Export pre-aggregated datasets for Tableau and write a data dictionary.

Design (decision log D27-D28):
  - Pre-aggregate in SQL; never ship 5M raw rows to Tableau
  - What-if tool compares rules and ML on the same test-period accounts
  - Auto-generate a data dictionary for handover

Prerequisites: scripts 01 to 09 have been run.
Run from project root:  python scripts/10_export_tableau.py
Outputs:                outputs/tableau/*.csv, docs/tableau_data_dictionary.md
"""
import csv
import shutil

import duckdb

from common import DB_PATH, OUT_DIR as ANALYSIS_DIR, ROOT

TAB_DIR = ROOT / "outputs" / "tableau"
DOC_PATH = ROOT / "docs" / "tableau_data_dictionary.md"
TEST_LO, TEST_HI = 20220906, 20220910

AMOUNT_BUCKET = """
    CASE WHEN f.amount_paid_usd <     100 THEN '1. < $100'
         WHEN f.amount_paid_usd <    1000 THEN '2. $100-1K'
         WHEN f.amount_paid_usd <    5000 THEN '3. $1K-5K'
         WHEN f.amount_paid_usd <    9000 THEN '4. $5K-9K'
         WHEN f.amount_paid_usd <   10000 THEN '5. $9K-10K'
         WHEN f.amount_paid_usd <   11000 THEN '6. $10K-11K'
         WHEN f.amount_paid_usd <   50000 THEN '7. $11K-50K'
         WHEN f.amount_paid_usd < 1000000 THEN '8. $50K-1M'
         ELSE                                  '9. > $1M' END"""

# name -> (grain, dashboard, purpose)
META = {
    "risk_cube.csv": ("date x hour x payment format x currency x amount band",
                      "1 Executive Risk", "Filterable laundering volume/rate by every risk dimension; includes tail flag"),
    "segment_risk.csv": ("one row per segment of each risk dimension",
                         "1 Executive Risk", "Laundering rate, 95% Wilson CI, lift, p-value, Bonferroni significance"),
    "kpi_headline.csv": ("single row", "1 Executive Risk", "Headline KPI tiles (analysis window, tail excluded)"),
    "typology_summary.csv": ("one row per typology", "2 Laundering Behaviour",
                             "Volume, value, AML stage, difficulty + behavioural fingerprint"),
    "account_feature_comparison.csv": ("one row per account feature", "2 Laundering Behaviour",
                                       "Laundering vs normal medians + univariate AUC"),
    "account_scores.csv": ("one row per account active in test period (Sep 6-10)", "3 Rules vs ML What-If",
                           "ML scores + ranks + rule flags + laundering group; drives the alert-budget slider"),
    "rule_performance.csv": ("one row per rule (full window)", "3 Rules vs ML What-If",
                             "Alerts, precision, recall, lift, cost per true positive"),
    "rule_threshold_sweep.csv": ("rule x threshold (full window)", "3 Rules vs ML What-If",
                                 "Precision/recall trade-off curves for R1, R2, R6"),
    "rule_typology_coverage.csv": ("rule x typology (full window)", "2 Laundering Behaviour",
                                   "Which typologies each rule catches"),
    "ml_vs_rules.csv": ("one row per rule (test period)", "3 Rules vs ML What-If",
                        "Head-to-head at identical alert budgets"),
    "ml_capacity_curve.csv": ("model x alert budget (test period)", "3 Rules vs ML What-If",
                              "Precision/recall of each model across alert budgets"),
    "ml_permutation_importance.csv": ("one row per feature (15-feature study)", "3 Rules vs ML What-If",
                                      "Test-set permutation importance vs gain importance"),
    "ml_bootstrap.csv": ("one row per headline metric", "3 Rules vs ML What-If",
                         "Bootstrap 95% CIs for PR-AUC and gain vs R3"),
    "ml_ablation.csv": ("one row per model variant", "3 Rules vs ML What-If",
                        "Ablation x seed stability results"),
}

COPY_FROM_ANALYSIS = [
    "account_feature_comparison.csv", "rule_performance.csv", "rule_threshold_sweep.csv",
    "rule_typology_coverage.csv", "ml_vs_rules.csv", "ml_capacity_curve.csv",
    "ml_permutation_importance.csv", "ml_bootstrap.csv", "ml_ablation.csv",
]

SEGMENT_FILES = {
    "Payment format": "risk_by_payment_format.csv",
    "Currency": "risk_by_currency.csv",
    "Hour of day": "risk_by_hour.csv",
    "Date": "risk_by_date.csv",
    "Weekend vs weekday": "risk_by_weekend.csv",
    "Cross-currency": "risk_by_cross_currency.csv",
    "Amount band": "risk_by_amount_bucket.csv",
}


def export(con, name: str, sql: str) -> None:
    con.execute(f"COPY ({sql}) TO '{TAB_DIR / name}' (HEADER, DELIMITER ',')")
    print(f"  wrote {name}")


def export_risk_cube(con):
    export(con, "risk_cube.csv", f"""
        SELECT d.date, d.day_name, d.is_weekend, d.is_tail_period,
               f.txn_hour,
               pf.payment_format, pf.channel_type,
               pc.currency AS payment_currency, pc.iso_code,
               {AMOUNT_BUCKET} AS amount_bucket,
               count(*)                         AS txns,
               sum(f.is_laundering)             AS laundering_txns,
               round(sum(f.amount_paid_usd), 2) AS usd_total,
               round(coalesce(sum(f.amount_paid_usd) FILTER (WHERE f.is_laundering = 1), 0), 2)
                                                AS usd_laundering
        FROM fact_transactions f
        JOIN dim_date d            USING (date_key)
        JOIN dim_payment_format pf USING (payment_format_key)
        JOIN dim_currency pc       ON f.payment_currency_key = pc.currency_key
        GROUP BY ALL
        ORDER BY ALL
    """)


def export_segment_risk():
    rows = []
    for dimension, fname in SEGMENT_FILES.items():
        path = ANALYSIS_DIR / fname
        if not path.exists():
            print(f"  WARNING: missing {fname} (run 05_analysis_risk.py)")
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append({"dimension": dimension, **r})
    with open(TAB_DIR / "segment_risk.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("  wrote segment_risk.csv")


def export_kpis(con):
    export(con, "kpi_headline.csv", """
        WITH w AS (
            SELECT f.*, pf.payment_format
            FROM fact_transactions f
            JOIN dim_date d USING (date_key)
            JOIN dim_payment_format pf USING (payment_format_key)
            WHERE NOT d.is_tail_period
        )
        SELECT
            (SELECT min(date) FROM dim_date WHERE NOT is_tail_period)     AS window_start,
            (SELECT max(date) FROM dim_date WHERE NOT is_tail_period)     AS window_end,
            count(*)                                                      AS transactions,
            sum(is_laundering)                                            AS laundering_transactions,
            round(100.0 * avg(is_laundering), 4)                          AS laundering_rate_pct,
            round(sum(amount_paid_usd) / 1e9, 2)                          AS volume_usd_bn,
            round(100.0 * sum(is_laundering) FILTER (WHERE payment_format = 'ACH')
                  / sum(is_laundering), 1)                                AS ach_share_of_laundering_pct,
            (SELECT count(*) FROM feat_account)                           AS accounts,
            (SELECT sum(label_laundering_account) FROM feat_account)      AS laundering_accounts,
            (SELECT count(*) FROM dim_bank)                               AS banks,
            (SELECT count(*) - 1 FROM dim_typology WHERE typology <> 'UNDOCUMENTED')
                                                                          AS documented_typologies
        FROM w
    """)


def export_typology(con):
    fp = ANALYSIS_DIR / "typology_fingerprint.csv"
    export(con, "typology_summary.csv", f"""
        WITH fp AS (SELECT * FROM read_csv_auto('{fp}'))
        SELECT t.typology, t.description, t.aml_stage, t.detection_difficulty,
               count(*)                                   AS txns,
               count(DISTINCT f.attempt_id)               AS attempts,
               round(median(f.amount_paid_usd), 2)        AS median_usd,
               round(sum(f.amount_paid_usd), 2)           AS total_usd,
               fp.sender_out_degree, fp.receiver_in_degree,
               fp.sender_pass_through, fp.receiver_pass_through, fp.sender_dwell_hours
        FROM fact_transactions f
        JOIN dim_typology t USING (typology_key)
        JOIN dim_date d     USING (date_key)
        LEFT JOIN fp        ON fp.typology = t.typology
        WHERE NOT d.is_tail_period
        GROUP BY ALL
        ORDER BY txns DESC
    """)


def export_account_scores(con):
    export(con, "account_scores.csv", f"""
        WITH lt AS (
            SELECT f.from_account_sk AS account_sk, t.typology
            FROM fact_transactions f JOIN dim_typology t USING (typology_key)
            WHERE f.date_key BETWEEN {TEST_LO} AND {TEST_HI} AND t.typology <> 'NONE'
            UNION ALL
            SELECT f.to_account_sk, t.typology
            FROM fact_transactions f JOIN dim_typology t USING (typology_key)
            WHERE f.date_key BETWEEN {TEST_LO} AND {TEST_HI} AND t.typology <> 'NONE'
        ),
        typ AS (
            SELECT account_sk,
                   max((typology <> 'UNDOCUMENTED')::INT) AS has_documented,
                   max((typology =  'UNDOCUMENTED')::INT) AS has_undocumented
            FROM lt GROUP BY 1
        )
        SELECT s.account_sk,
               a.bank_id,
               s.label AS is_laundering_account,
               CASE WHEN typ.has_documented = 1   THEN 'Documented typology'
                    WHEN typ.has_undocumented = 1 THEN 'Undocumented'
                    ELSE 'Not laundering' END      AS laundering_group,
               round(s.score_lightgbm, 6)          AS score_lightgbm,
               s.rank_lightgbm,
               round(s.score_logreg, 6)            AS score_logreg,
               row_number() OVER (ORDER BY s.score_logreg DESC) AS rank_logreg,
               s.* EXCLUDE (account_sk, label, score_lightgbm, rank_lightgbm, score_logreg),
               ft.in_degree, ft.txns_in, ft.usd_in, ft.ach_share_out, ft.pass_through_ratio
        FROM ml_scores_test s
        JOIN feat_account_test ft USING (account_sk)
        LEFT JOIN dim_account a   USING (account_sk)
        LEFT JOIN typ             USING (account_sk)
        ORDER BY s.rank_lightgbm
    """)


def copy_results():
    for fname in COPY_FROM_ANALYSIS:
        src = ANALYSIS_DIR / fname
        if src.exists():
            shutil.copy(src, TAB_DIR / fname)
            print(f"  wrote {fname}")
        else:
            print(f"  WARNING: missing {fname}")


def write_dictionary(con):
    lines = [
        "# Tableau Data Dictionary",
        "",
        "Generated by `scripts/10_export_tableau.py`. All files live in `outputs/tableau/`.",
        "",
        "| File | Rows | Grain | Dashboard | Purpose |",
        "|---|---:|---|---|---|",
    ]
    details = []
    for fname, (grain, dash, purpose) in META.items():
        path = TAB_DIR / fname
        if not path.exists():
            continue
        n = con.execute(f"SELECT count(*) FROM read_csv_auto('{path}')").fetchone()[0]
        cols = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{path}')").fetchall()
        lines.append(f"| `{fname}` | {n:,} | {grain} | {dash} | {purpose} |")
        details.append(f"\n## `{fname}`\n")
        details.append("| Column | Type |\n|---|---|")
        details += [f"| `{c[0]}` | {c[1]} |" for c in cols]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines + details) + "\n")
    print(f"  wrote {DOC_PATH.relative_to(ROOT)}")


def main():
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    print("Exporting Tableau datasets ...")
    export_risk_cube(con)
    export_segment_risk()
    export_kpis(con)
    export_typology(con)
    export_account_scores(con)
    copy_results()
    print("Writing data dictionary ...")
    write_dictionary(con)
    con.close()
    print(f"\nTableau exports ready in {TAB_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

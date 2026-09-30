"""
Model vs rules: does a model catch more laundering at the same analyst workload?

Design (see docs/decision_log.md D19-D26):
  - Out-of-time split: train on Sep 1-5, test on Sep 6-10
  - Features recomputed separately per period (no future information)
  - Models: logistic regression (baseline) + LightGBM (final 14-feature set, D26)
  - Rules re-evaluated on the same test period
  - Metrics: PR-AUC, ROC-AUC, precision/recall at each rule's alert budget

Run from project root:  python scripts/08_ml_vs_rules.py
Outputs:                ml_scores_test table in aml.duckdb, outputs/analysis/ml_*.csv
"""
import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import (DB_PATH, FEATURES, LABEL, LGBM_PARAMS, OUT_DIR, ROOT,
                    apply_rules, topk_hits, write_csv)

SEED = 42
PERIODS = {"train": (20220901, 20220905), "test": (20220906, 20220910)}


# ------------------------------------------------------------ features per period
def build_period_features(con, name: str, lo: int, hi: int) -> None:
    """Same features as 06_account_behaviour.py, computed within one period only."""
    con.execute(f"""
        CREATE OR REPLACE TABLE feat_account_{name} AS
        WITH f AS (
            SELECT f.*, pf.payment_format, d.is_weekend
            FROM fact_transactions f
            JOIN dim_date d            USING (date_key)
            JOIN dim_payment_format pf USING (payment_format_key)
            WHERE NOT d.is_tail_period AND f.date_key BETWEEN {lo} AND {hi}
        ),
        outg AS (
            SELECT from_account_sk AS account_sk,
                   count(*) AS txns_out,
                   count(DISTINCT to_account_sk) FILTER (WHERE NOT is_self_transfer) AS out_degree,
                   sum(amount_paid_usd)::DOUBLE AS usd_out,
                   median(amount_paid_usd)::DOUBLE AS median_amt_out,
                   avg((payment_format = 'ACH')::INT) AS ach_share_out,
                   count(DISTINCT payment_format) AS n_formats_out,
                   avg((amount_paid_usd BETWEEN 1000 AND 50000)::INT) AS midband_share_out,
                   avg(is_weekend::INT) AS weekend_share_out,
                   min(ts) AS first_out, max(ts) AS last_out,
                   max(is_laundering) AS lbl_out
            FROM f GROUP BY 1
        ),
        inc AS (
            SELECT to_account_sk AS account_sk,
                   count(*) AS txns_in,
                   count(DISTINCT from_account_sk) FILTER (WHERE NOT is_self_transfer) AS in_degree,
                   sum(amount_received_usd)::DOUBLE AS usd_in,
                   min(ts) AS first_in, max(ts) AS last_in,
                   max(is_laundering) AS lbl_in
            FROM f GROUP BY 1
        ),
        base AS (
            SELECT coalesce(o.account_sk, i.account_sk) AS account_sk,
                   o.* EXCLUDE (account_sk), i.* EXCLUDE (account_sk)
            FROM outg o FULL OUTER JOIN inc i ON o.account_sk = i.account_sk
        ),
        burst AS (
            SELECT from_account_sk AS account_sk, max(n) AS max_txns_out_per_hour
            FROM (SELECT from_account_sk, date_trunc('hour', ts) AS h, count(*) AS n
                  FROM f GROUP BY 1, 2)
            GROUP BY 1
        ),
        dwell AS (
            SELECT i.account_sk,
                   date_diff('minute', i.first_in, min(o.ts)) / 60.0 AS hours_in_to_out
            FROM inc i
            JOIN f o ON o.from_account_sk = i.account_sk
                    AND o.ts >= i.first_in AND NOT o.is_self_transfer
            GROUP BY i.account_sk, i.first_in
        )
        SELECT b.account_sk,
               coalesce(txns_out, 0) AS txns_out, coalesce(txns_in, 0) AS txns_in,
               coalesce(out_degree, 0) AS out_degree, coalesce(in_degree, 0) AS in_degree,
               coalesce(usd_out, 0) AS usd_out, coalesce(usd_in, 0) AS usd_in,
               CASE WHEN greatest(coalesce(usd_in, 0), coalesce(usd_out, 0)) > 0
                    THEN least(coalesce(usd_in, 0), coalesce(usd_out, 0))
                       / greatest(coalesce(usd_in, 0), coalesce(usd_out, 0))
                    ELSE 0 END AS pass_through_ratio,
               median_amt_out, ach_share_out, n_formats_out,
               midband_share_out, weekend_share_out,
               coalesce(bu.max_txns_out_per_hour, 0) AS max_txns_out_per_hour,
               dw.hours_in_to_out,
               date_diff('minute',
                         least(coalesce(first_out, first_in), coalesce(first_in, first_out)),
                         greatest(coalesce(last_out, last_in), coalesce(last_in, last_out))
                        ) / 60.0 AS active_span_hours,
               greatest(coalesce(lbl_out, 0), coalesce(lbl_in, 0)) AS {LABEL}
        FROM base b
        LEFT JOIN burst bu USING (account_sk)
        LEFT JOIN dwell dw USING (account_sk)
    """)


def main():
    con = duckdb.connect(str(DB_PATH))

    # 1. Features per period
    for name, (lo, hi) in PERIODS.items():
        print(f"Building features for {name} period {lo}-{hi} ...")
        build_period_features(con, name, lo, hi)
    train = con.execute("SELECT * FROM feat_account_train").df()
    test = con.execute("SELECT * FROM feat_account_test").df()

    X_tr, y_tr = train[FEATURES].astype(float), train[LABEL].astype(int).to_numpy()
    X_te, y_te = test[FEATURES].astype(float), test[LABEL].astype(int).to_numpy()
    print(f"\nTRAIN: {len(train):,} accounts, {y_tr.sum():,} laundering ({100 * y_tr.mean():.3f}%)")
    print(f"TEST : {len(test):,} accounts, {y_te.sum():,} laundering ({100 * y_te.mean():.3f}%)")

    # 2. Models
    lr = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        FunctionTransformer(np.log1p),          # all features are >= 0; tames skew
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    )
    lgb = LGBMClassifier(**LGBM_PARAMS, random_state=SEED)
    print("Training logistic regression ...")
    lr.fit(X_tr, y_tr)
    print("Training LightGBM ...")
    lgb.fit(X_tr, y_tr)

    s_lr = lr.predict_proba(X_te)[:, 1]
    s_lgb = lgb.predict_proba(X_te)[:, 1]

    # 3. Threshold-free metrics
    print("\n=== 1. MODEL QUALITY ON TEST (Sep 6-10) ===")
    print(f"  {'model':<22}{'PR-AUC':>9}{'ROC-AUC':>9}")
    print(f"  {'random (baseline)':<22}{y_te.mean():>9.4f}{0.5:>9.3f}")
    for name, s in [("logistic_regression", s_lr), ("lightgbm", s_lgb)]:
        print(f"  {name:<22}{average_precision_score(y_te, s):>9.4f}{roc_auc_score(y_te, s):>9.3f}")

    # 4. Head-to-head at each rule's alert budget
    rules = apply_rules(test)
    pos = int(y_te.sum())
    rows = []
    print("\n=== 2. MODEL VS RULES AT EQUAL ALERT BUDGETS (budget = each rule's alert count) ===")
    print(f"  {'rule':<24}{'alerts':>9}{'rule TP':>9}{'LR TP':>8}{'LGBM TP':>9}"
          f"{'rule prec%':>11}{'LGBM prec%':>11}{'rule rec%':>10}{'LGBM rec%':>10}{'gain':>7}")
    for rname, flags in rules.items():
        k = int(flags.sum())
        tp_rule = int(y_te[flags.to_numpy()].sum())
        tp_lr, tp_lgb = topk_hits(s_lr, y_te, k), topk_hits(s_lgb, y_te, k)
        gain = tp_lgb / tp_rule if tp_rule else float("nan")
        rows.append({"rule": rname, "alert_budget": k,
                     "rule_tp": tp_rule, "lr_tp": tp_lr, "lgbm_tp": tp_lgb,
                     "rule_precision_pct": 100 * tp_rule / k if k else 0,
                     "lgbm_precision_pct": 100 * tp_lgb / k if k else 0,
                     "rule_recall_pct": 100 * tp_rule / pos,
                     "lgbm_recall_pct": 100 * tp_lgb / pos,
                     "lgbm_vs_rule_gain": gain})
        r = rows[-1]
        print(f"  {rname:<24}{k:>9,}{tp_rule:>9,}{tp_lr:>8,}{tp_lgb:>9,}"
              f"{r['rule_precision_pct']:>11.2f}{r['lgbm_precision_pct']:>11.2f}"
              f"{r['rule_recall_pct']:>10.1f}{r['lgbm_recall_pct']:>10.1f}{gain:>6.2f}x")
    write_csv("ml_vs_rules.csv", rows)

    # 5. Capacity curve for Tableau (precision/recall at many alert budgets)
    curve = []
    for k in [250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000,
              30000, 50000, 75000, 100000, 150000, 200000]:
        for name, s in [("logistic_regression", s_lr), ("lightgbm", s_lgb)]:
            tp = topk_hits(s, y_te, k)
            curve.append({"model": name, "alert_budget": k, "true_positives": tp,
                          "precision_pct": 100 * tp / k, "recall_pct": 100 * tp / pos})
    write_csv("ml_capacity_curve.csv", curve)

    # 6. Typology coverage at R3's budget: documented vs UNDOCUMENTED
    lo, hi = PERIODS["test"]
    typ = con.execute(f"""
        SELECT DISTINCT account_sk, typology FROM (
            SELECT f.from_account_sk AS account_sk, t.typology
            FROM fact_transactions f JOIN dim_typology t USING (typology_key)
            WHERE f.date_key BETWEEN {lo} AND {hi} AND t.typology <> 'NONE'
            UNION ALL
            SELECT f.to_account_sk, t.typology
            FROM fact_transactions f JOIN dim_typology t USING (typology_key)
            WHERE f.date_key BETWEEN {lo} AND {hi} AND t.typology <> 'NONE'
        )
    """).df()
    k_r3 = int(rules["R3_ach_concentration"].sum())
    lgb_flag = pd.Series(False, index=test.index)
    lgb_flag.iloc[np.argsort(-s_lgb, kind="stable")[:k_r3]] = True
    flagged = {"R3_ach_concentration": set(test.account_sk[rules["R3_ach_concentration"]]),
               "lightgbm_at_R3_budget": set(test.account_sk[lgb_flag])}
    doc = set(typ.account_sk[typ.typology != "UNDOCUMENTED"])
    und = set(typ.account_sk[typ.typology == "UNDOCUMENTED"])
    print(f"\n=== 3. TYPOLOGY COVERAGE at R3 budget ({k_r3:,} alerts) ===")
    print(f"  {'method':<24}{'documented rec%':>17}{'UNDOCUMENTED rec%':>20}")
    for name, s in flagged.items():
        print(f"  {name:<24}{100 * len(s & doc) / max(len(doc), 1):>17.1f}"
              f"{100 * len(s & und) / max(len(und), 1):>20.1f}")

    # 7. Feature importance (gain)
    gain = lgb.booster_.feature_importance(importance_type="gain")
    imp = sorted(zip(FEATURES, 100 * gain / gain.sum()), key=lambda t: -t[1])
    print("\n=== 4. LIGHTGBM FEATURE IMPORTANCE (gain %) ===")
    for f, g in imp:
        print(f"  {f:<24}{g:>6.1f}%  {'#' * int(g / 2)}")
    write_csv("ml_feature_importance.csv",
              [{"feature": f, "gain_pct": g} for f, g in imp])

    # 8. Persist scores for Tableau
    out = pd.DataFrame({
        "account_sk": test.account_sk, "label": y_te,
        "score_lightgbm": s_lgb, "score_logreg": s_lr,
        "rank_lightgbm": pd.Series(s_lgb).rank(ascending=False, method="first").astype(int),
        **{f"rule_{k}": v.to_numpy() for k, v in rules.items()},
    })
    con.register("scores_df", out)
    con.execute("CREATE OR REPLACE TABLE ml_scores_test AS SELECT * FROM scores_df")
    con.close()
    print(f"\nML vs rules complete; scores in ml_scores_test, tables in {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

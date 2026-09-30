"""
Robustness checks for the model result.

  1. Permutation importance on the test set (unlike gain importance, not biased toward continuous features)
  2. Ablation: retrain without suspected artifact features
  3. Seed stability: 3 seeds per variant -> mean and standard deviation
  4. Bootstrap 95% CIs for PR-AUC and the LightGBM-vs-R3 gain

Prerequisite: run 08_ml_vs_rules.py first (builds feat_account_train / _test).
Run from project root:  python scripts/09_robustness.py      (~3-6 minutes)
Outputs:                outputs/analysis/ml_permutation_importance.csv,
                        ml_ablation.csv, ml_bootstrap.csv
"""
import time

import duckdb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from common import (DB_PATH, FEATURES_ALL as FEATURES, LABEL, LGBM_PARAMS, OUT_DIR,
                    apply_rules, topk_hits, write_csv)
# The robustness study uses the full 15-feature set; the final model's
# feature set is common.FEATURES.

SEEDS = [42, 7, 2026]
N_BOOT = 200

ABLATIONS = {
    "full_model":           FEATURES,
    "no_active_span":       [f for f in FEATURES if f != "active_span_hours"],
    "no_time_window_feats": [f for f in FEATURES if f not in
                             ("active_span_hours", "hours_in_to_out",
                              "weekend_share_out", "max_txns_out_per_hour")],
    "top5_univariate_only": ["in_degree", "usd_in", "ach_share_out", "txns_in", "n_formats_out"],
}


def train(X, y, seed):
    return LGBMClassifier(**LGBM_PARAMS, random_state=seed).fit(X, y)


def main():
    t0 = time.time()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    train_df = con.execute("SELECT * FROM feat_account_train").df()
    test_df = con.execute("SELECT * FROM feat_account_test").df()
    con.close()

    y_tr = train_df[LABEL].astype(int).to_numpy()
    y_te = test_df[LABEL].astype(int).to_numpy()
    r3 = apply_rules(test_df)["R3_ach_concentration"].to_numpy()
    k_r3, r3_tp = int(r3.sum()), int(y_te[r3].sum())

    # ---------------------------------------------- 2 + 3. ablation x seeds
    print(f"=== ABLATION x SEED STABILITY  (R3 budget = {k_r3:,} alerts, R3 catches {r3_tp}) ===")
    print(f"  {'variant':<24}{'#feat':>6}{'PR-AUC mean':>13}{'std':>8}{'TP@R3 mean':>12}{'gain vs R3':>12}")
    ablation_rows, full_model, full_scores = [], None, None
    for name, feats in ABLATIONS.items():
        aps, tps = [], []
        for seed in SEEDS:
            m = train(train_df[feats].astype(float), y_tr, seed)
            s = m.predict_proba(test_df[feats].astype(float))[:, 1]
            aps.append(average_precision_score(y_te, s))
            tps.append(topk_hits(s, y_te, k_r3))
            if name == "full_model" and seed == SEEDS[0]:
                full_model, full_scores = m, s
        row = {"variant": name, "n_features": len(feats),
               "pr_auc_mean": float(np.mean(aps)), "pr_auc_std": float(np.std(aps)),
               "tp_at_r3_budget_mean": float(np.mean(tps)),
               "gain_vs_r3": float(np.mean(tps)) / r3_tp}
        ablation_rows.append(row)
        print(f"  {name:<24}{len(feats):>6}{row['pr_auc_mean']:>13.4f}{row['pr_auc_std']:>8.4f}"
              f"{row['tp_at_r3_budget_mean']:>12.0f}{row['gain_vs_r3']:>11.2f}x")
    write_csv("ml_ablation.csv", ablation_rows)

    # ---------------------------------------------- 1. permutation importance
    print("\n=== PERMUTATION IMPORTANCE on test set (drop in PR-AUC when a feature is shuffled) ===")
    X_te = test_df[FEATURES].astype(float)
    perm = permutation_importance(full_model, X_te, y_te, scoring="average_precision",
                                  n_repeats=5, random_state=42, n_jobs=-1)
    gain = full_model.booster_.feature_importance(importance_type="gain")
    gain_pct = 100 * gain / gain.sum()
    perm_rows = sorted(
        [{"feature": f, "pr_auc_drop": float(perm.importances_mean[i]),
          "pr_auc_drop_std": float(perm.importances_std[i]), "gain_pct": float(gain_pct[i])}
         for i, f in enumerate(FEATURES)],
        key=lambda r: -r["pr_auc_drop"])
    print(f"  {'feature':<24}{'PR-AUC drop':>12}{'std':>9}{'gain %':>9}")
    for r in perm_rows:
        print(f"  {r['feature']:<24}{r['pr_auc_drop']:>12.4f}{r['pr_auc_drop_std']:>9.4f}{r['gain_pct']:>9.1f}")
    write_csv("ml_permutation_importance.csv", perm_rows)

    # ---------------------------------------------- 4. bootstrap CIs
    print(f"\n=== BOOTSTRAP 95% CIs ({N_BOOT} resamples of test accounts) ===")
    rng = np.random.default_rng(42)
    n = len(y_te)
    boot_ap, boot_gain = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        y_b, s_b, r3_b = y_te[idx], full_scores[idx], r3[idx]
        boot_ap.append(average_precision_score(y_b, s_b))
        rule_tp_b = y_b[r3_b].sum()
        if rule_tp_b > 0:
            boot_gain.append(topk_hits(s_b, y_b, int(r3_b.sum())) / rule_tp_b)
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    ap_lo, ap_hi = ci(boot_ap)
    g_lo, g_hi = ci(boot_gain)
    print(f"  LightGBM PR-AUC            {np.mean(boot_ap):.4f}   95% CI [{ap_lo:.4f} - {ap_hi:.4f}]")
    print(f"  Gain vs R3 at same budget  {np.mean(boot_gain):.2f}x   95% CI [{g_lo:.2f}x - {g_hi:.2f}x]")
    print("  (Caveat: accounts in the same laundering scheme are correlated, so true CIs are "
          "somewhat wider.)")
    write_csv("ml_bootstrap.csv", [
        {"metric": "lightgbm_pr_auc", "mean": float(np.mean(boot_ap)), "ci_low": ap_lo, "ci_high": ap_hi},
        {"metric": "gain_vs_r3", "mean": float(np.mean(boot_gain)), "ci_low": g_lo, "ci_high": g_hi},
    ])

    print(f"\nRobustness checks complete in {time.time() - t0:.0f}s; tables in "
          f"{OUT_DIR.name}/")


if __name__ == "__main__":
    main()

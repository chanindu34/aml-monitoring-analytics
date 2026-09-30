"""
Rule simulation: how well do standard AML rules perform, and at which thresholds?

Account-level rule simulation on feat_account (analysis window only):
  1. KPIs per rule + union: alerts, alert rate, precision, recall, lift, cost per true positive
  2. Typology coverage: documented vs UNDOCUMENTED recall per rule
  3. Threshold sweep for R1 / R2 / R6  -> data for the Tableau what-if tool

Run from project root:  python scripts/07_rule_simulation.py
Outputs:                outputs/analysis/rule_*.csv
"""
import csv
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "aml.duckdb"
OUT_DIR = ROOT / "outputs" / "analysis"

# Business assumptions (exposed as Tableau parameters)
MINUTES_PER_ALERT = 30
ANALYST_COST_PER_HOUR = 40.0
COST_PER_ALERT = MINUTES_PER_ALERT / 60 * ANALYST_COST_PER_HOUR   # $20

LABEL = "label_laundering_account"

RULES = {
    "R1_high_in_degree":     ("in_degree >= 5",
                              "Receives from 5+ distinct counterparties"),
    "R2_high_out_degree":    ("out_degree >= 5",
                              "Sends to 5+ distinct counterparties"),
    "R3_ach_concentration":  ("ach_share_out >= 0.5 AND txns_out >= 3",
                              ">=50% of outgoing txns via ACH (min 3 txns)"),
    "R4_midband_volume":     ("midband_share_out >= 0.5 AND usd_out >= 25000",
                              ">=50% of txns in $1K-50K band and >=$25K out"),
    "R5_multi_format":       ("n_formats_out >= 3",
                              "Uses 3+ payment formats"),
    "R6_large_inflow":       ("usd_in >= 100000",
                              "Receives >= $100K in the window"),
    "R7_rapid_pass_through": ("hours_in_to_out <= 2 AND pass_through_ratio >= 0.8",
                              "Control: classic velocity rule (hypothesis rejected in 06_account_behaviour.py)"),
}
UNION_NAME = "ANY_RULE_R1_to_R6"
UNION_EXPR = " OR ".join(f"({RULES[r][0]})" for r in RULES if not r.startswith("R7"))

SWEEPS = {
    "R1_high_in_degree":  ("in_degree >= {t}",  list(range(1, 31))),
    "R2_high_out_degree": ("out_degree >= {t}", list(range(1, 31))),
    "R6_large_inflow":    ("usd_in >= {t}",     [1e3, 2.5e3, 5e3, 1e4, 2.5e4, 5e4, 1e5,
                                                 2.5e5, 5e5, 1e6, 2.5e6, 5e6, 1e7]),
}


def q(con, sql):
    return con.execute(sql).fetchone()


def kpis(con, expr: str, total_pos: int, total_acc: int) -> dict:
    alerts, tp = q(con, f"""
        SELECT count(*) FILTER (WHERE coalesce(({expr}), false)),
               count(*) FILTER (WHERE coalesce(({expr}), false) AND {LABEL} = 1)
        FROM feat_account
    """)
    base = total_pos / total_acc
    precision = tp / alerts if alerts else 0.0
    return {
        "alerts": alerts,
        "alert_rate_pct": 100 * alerts / total_acc,
        "true_positives": tp,
        "precision_pct": 100 * precision,
        "false_positive_rate_pct": 100 * (1 - precision) if alerts else 0.0,  # AML definition
        "recall_pct": 100 * tp / total_pos,
        "lift_vs_random": precision / base if alerts else 0.0,
        "analyst_cost_usd": alerts * COST_PER_ALERT,
        "cost_per_true_positive_usd": alerts * COST_PER_ALERT / tp if tp else None,
    }


def write_csv(name, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def rule_performance(con, total_pos, total_acc):
    rows = []
    for name, (expr, desc) in RULES.items():
        rows.append({"rule": name, "logic": expr, "description": desc,
                     **kpis(con, expr, total_pos, total_acc)})
    rows.append({"rule": UNION_NAME, "logic": "R1 OR ... OR R6",
                 "description": "Alert if any production rule fires",
                 **kpis(con, UNION_EXPR, total_pos, total_acc)})

    print(f"\n=== 1. RULE PERFORMANCE (account level)  |  cost/alert = ${COST_PER_ALERT:.0f}"
          f"  |  random precision = {100 * total_pos / total_acc:.2f}% ===")
    print(f"  {'rule':<24}{'alerts':>9}{'alert%':>8}{'TP':>7}{'prec%':>8}{'recall%':>9}"
          f"{'lift':>7}{'cost/TP':>10}")
    for r in rows:
        cpt = "n/a" if r["cost_per_true_positive_usd"] is None else f"${r['cost_per_true_positive_usd']:,.0f}"
        print(f"  {r['rule']:<24}{r['alerts']:>9,}{r['alert_rate_pct']:>8.2f}{r['true_positives']:>7,}"
              f"{r['precision_pct']:>8.2f}{r['recall_pct']:>9.2f}{r['lift_vs_random']:>6.1f}x{cpt:>10}")
    write_csv("rule_performance.csv", rows)


def typology_coverage(con):
    # Accounts involved (as sender or receiver) in each typology, analysis window only
    con.execute("""
        CREATE OR REPLACE TEMP TABLE acct_typology AS
        WITH lt AS (
            SELECT f.from_account_sk, f.to_account_sk, t.typology
            FROM fact_transactions f
            JOIN dim_typology t USING (typology_key)
            JOIN dim_date d     USING (date_key)
            WHERE NOT d.is_tail_period AND t.typology <> 'NONE'
        )
        SELECT DISTINCT from_account_sk AS account_sk, typology FROM lt
        UNION
        SELECT DISTINCT to_account_sk, typology FROM lt
    """)
    all_rules = {**{k: v[0] for k, v in RULES.items()}, UNION_NAME: UNION_EXPR}
    rows = []
    for name, expr in all_rules.items():
        for typology, n_acc, caught in con.execute(f"""
            SELECT a.typology, count(*) AS n_acc,
                   count(*) FILTER (WHERE coalesce(({expr}), false)) AS caught
            FROM acct_typology a JOIN feat_account USING (account_sk)
            GROUP BY 1 ORDER BY 1
        """).fetchall():
            rows.append({"rule": name, "typology": typology, "accounts": n_acc,
                         "caught": caught, "recall_pct": 100 * caught / n_acc})
    write_csv("rule_typology_coverage.csv", rows)

    print("\n=== 2. RECALL BY TYPOLOGY GROUP: documented vs UNDOCUMENTED ===")
    print(f"  {'rule':<24}{'documented recall%':>20}{'UNDOCUMENTED recall%':>22}")
    for name in all_rules:
        doc = [r for r in rows if r["rule"] == name and r["typology"] != "UNDOCUMENTED"]
        und = [r for r in rows if r["rule"] == name and r["typology"] == "UNDOCUMENTED"]
        d_rec = 100 * sum(r["caught"] for r in doc) / sum(r["accounts"] for r in doc)
        u_rec = und[0]["recall_pct"] if und else 0.0
        print(f"  {name:<24}{d_rec:>20.1f}{u_rec:>22.1f}")


def threshold_sweep(con, total_pos, total_acc):
    rows = []
    for rule, (template, thresholds) in SWEEPS.items():
        for t in thresholds:
            rows.append({"rule": rule, "threshold": t,
                         **kpis(con, template.format(t=t), total_pos, total_acc)})
    write_csv("rule_threshold_sweep.csv", rows)

    print("\n=== 3. THRESHOLD SWEEP: R1 high in-degree (excerpt) ===")
    print(f"  {'threshold':>10}{'alerts':>10}{'prec%':>8}{'recall%':>9}{'cost/TP':>10}")
    for r in rows:
        if r["rule"] == "R1_high_in_degree" and r["threshold"] in (1, 2, 3, 4, 5, 7, 10, 15, 20, 30):
            cpt = "n/a" if r["cost_per_true_positive_usd"] is None else f"${r['cost_per_true_positive_usd']:,.0f}"
            print(f"  {r['threshold']:>10}{r['alerts']:>10,}{r['precision_pct']:>8.2f}"
                  f"{r['recall_pct']:>9.2f}{cpt:>10}")
    print(f"  (full sweep for R1, R2, R6 saved: {len(rows)} rows)")


def main():
    con = duckdb.connect(str(DB_PATH))
    total_acc, total_pos = q(con, f"SELECT count(*), sum({LABEL}) FROM feat_account")
    print(f"Accounts: {total_acc:,}  |  laundering-involved: {total_pos:,}")
    rule_performance(con, total_pos, total_acc)
    typology_coverage(con)
    threshold_sweep(con, total_pos, total_acc)
    con.close()
    print(f"\nRule simulation complete; tables saved to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

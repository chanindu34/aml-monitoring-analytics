# Decision Log

Each entry records what was chosen, what was rejected, the reason and the supporting evidence.

## Data and environment

### D01. Dataset: IBM AML synthetic transactions, HI-Small variant
- Rejected: real bank data (confidential), SAML-D (smaller, fewer typology labels), HI-Large (about 180M rows; the full dataset is 8.2 GB, too large to work with locally).
- Reason: the only public AML dataset with both ground-truth laundering labels and a typology file. Synthetic data also has complete labels, which real banks never have for missed laundering.
- Evidence: 5,078,345 transactions, 5,177 labelled as laundering, 8 typologies, CDLA-Sharing 1.0 licence.

### D02. Ingestion through the Kaggle API
- Rejected: manual browser download (not reproducible), web scraping (against the site terms, and unnecessary when an API exists).
- Reason: one reproducible command. The API token is stored outside the repository, and raw data is kept unmodified and out of git.

### D03. Python venv for the environment
- Rejected: global Python (dependency conflicts), conda (heavier than needed without C or GPU dependencies), Docker (a deployment tool, more than this project needs).
- Reason: built in, lightweight, reproducible through requirements.txt.

### D04. DuckDB as the analytics engine
- Rejected: pandas only (5M rows in memory, slow aggregations), PostgreSQL or Supabase (transactional databases, slower for large scans), SQLite (row-based storage), Qdrant (a vector database for similarity search, a different problem), BigQuery or Snowflake (cost and setup not justified).
- Reason: an in-process columnar engine. Aggregations over millions of rows run in seconds, with full SQL, a single file and no server.

## Data engineering

### D05. Explicit schema on load
- Rejected: automatic type inference.
- Reason: the raw file has two columns both named `Account`, and bank IDs with leading zeros. Read as integers, `010` becomes `10` and joins fail silently.
- Evidence: columns renamed to `from_account` and `to_account`; IDs read as text.

### D06. Money stored as DECIMAL(24,8)
- Rejected: FLOAT (binary rounding errors), DECIMAL(18,2) (the first version, which caused a bug).
- Reason: two decimal places rounded Bitcoin amounts (0.025852 BTC became 0.03, a 16% error). This created 389 false duplicates and removed Bitcoin from the FX analysis. Eight decimal places fixed it, and also reduced FX rate noise for every currency.

### D07. Bronze, silver and gold layers
- Rejected: a single clean-and-load script.
- Reason: raw data stays unmodified and can always be rebuilt. Each layer has one job: copy, clean, model.

### D08. Investigate before cleaning; flag rather than delete
- Rejected: dropping duplicates and self-transfers without checking them.
- Reason: 9 true duplicates were all non-laundering, so they are flagged and excluded from the gold layer. The 591K self-transfers include 11 laundering transactions, so they are kept and flagged.

### D09. FX rates derived from the data
- Rejected: external market rates (the synthetic data may use its own rates), a daily FX table (not needed).
- Reason: cross-currency transactions reveal the simulator's rates. The median per currency resists outliers. Rates varied by less than 0.1%, so one fixed rate per currency is supported by the data.

### D10. Composite account key (bank ID plus account ID)
- Rejected: account ID alone.
- Reason: a row-level check found no collisions, but a dataset-wide check found 8 account IDs used at more than one bank. Treating them as one account would merge different customers.
- Evidence: dim_account has 515,088 rows, which is 515,080 raw IDs plus 8.

### D11. Star schema
- Rejected: one wide flat table (duplicated data, slow in BI tools), a snowflake schema (extra joins).
- Reason: a declared grain of one row per unique transaction, with every dimension one join from the fact table. This maps directly to Tableau relationships.
- Details: account and currency dimensions are reused for sender and receiver, and for paid and received currency. Placeholder rows (NONE, UNDOCUMENTED) keep foreign keys from being NULL. Columns derived from the label start with `label_` and are used in dashboards only, never as model features.
- Evidence: 8 checks on row counts, USD totals, laundering counts and referential integrity.

### D12. Tail period flagged with a volume rule
- Rejected: keeping it (it inflates every rate), hardcoding a date (fragile).
- Reason: after Sep 10, volume falls to 11-396 transactions a day with 57-73% laundering. The simulation ended, but laundering chains already in progress kept completing. Rule: a day with less than 1% of the median daily volume is excluded.
- Evidence: 655 laundering transactions (12.7%) fall in the tail. All findings were rerun without it and held. Excluding it also stops the model learning dates.

## Analysis

### D13. Wilson confidence intervals, lift and Bonferroni correction
- Rejected: raw rates only, normal-approximation intervals (unreliable near 0%, and the base rate is 0.09%), uncorrected p-values.
- Reason: rare-event rates need stable intervals, lift is easy for managers to read, and testing 15-24 segments at once produces false positives unless alpha is divided by the number of tests.
- Note: statistical and practical significance differ. The Euro result is significant but only 1.16x.

### D14. Medians reported alongside totals
- Rejected: means and sums alone.
- Reason: extreme outliers (the largest single transaction is about $26.6B). An early claim that undocumented laundering held 87% of laundered value was overturned by medians ($2,380 median against an $11.9M mean).

### D15. Monitoring at account level
- Rejected: alerting on individual transactions.
- Reason: analysts investigate customers, not single payments. Laundering accounts are 1.14% of accounts against 0.089% of transactions, a much less imbalanced problem.

### D16. Univariate AUC to rank features
- Rejected: comparing means, medians or correlations.
- Reason: AUC is rank-based, so outliers do not distort it, and it has a direct meaning: the probability that a laundering account scores higher than a normal one. It also picks up differences in the tails that medians miss (out_degree has a median of 1 in both groups but an AUC of 0.654).
- Check: fan-out senders have a median out-degree of 11 and fan-in receivers a median in-degree of 12, which confirms the feature logic against known patterns.

## Rules

### D17. Rules based on the findings, plus a control rule
- Rejected: generic textbook rules.
- Reason: each rule maps to a finding (ACH at 7.2x, in-degree AUC 0.73, the $1K-50K band at up to 3.6x). A classic velocity rule (R7) was included as a control after the data rejected the idea that launderers move money quickly. It caught 1.6% of laundering accounts.

### D18. Cost model: $20 per alert (30 minutes at $40 per hour)
- Reason: turns precision into cost per case found. The figures are assumptions, so D3 lets the viewer change them.
- Evidence: the six rules combined reach 81% recall but alert on 46.5% of accounts, with a 98% false positive rate and about $4.8M of analyst time over 10 days.

## Machine learning

### D19. Out-of-time split: train on Sep 1-5, test on Sep 6-10, features rebuilt per period
- Rejected: a random account split (accounts in the same laundering scheme land on both sides, and features include future activity), a group split (still uses future-period features).
- Reason: matches how a bank would use the model, training on past data and scoring current activity. Rules are evaluated on the same test period for a fair comparison.
- Trade-off: scores are lower than a leaky split would give.

### D20. Metrics: PR-AUC and results at a fixed alert budget
- Rejected: accuracy (a model that never flags anything scores 99.9%), ROC-AUC alone (looks good at 1% prevalence even when precision is poor).
- Reason: PR-AUC focuses on the rare class, and the alert budget comparison answers the business question directly: at the same workload, which approach catches more laundering?

### D21. Models: logistic regression baseline, then LightGBM
- Rejected: deep learning (not needed for 15 tabular features, harder to explain), skipping the baseline.
- Reason: a simple baseline first, then gradient boosting, the usual choice for tabular fraud and AML data. It handles skewed and missing values without scaling.
- Imbalance: class weights, and no resampling of the test set, which keeps its real prevalence.
- Leakage check: the code fails if any `label_` column is used as a feature.

### D22. Permutation importance on the test set
- Rejected: relying on LightGBM gain importance.
- Reason: gain importance is computed on training data and favours continuous features with many distinct values. `active_span_hours` ranked first by gain (22%) but only seventh on its own (AUC 0.655). Permutation importance measures the actual drop in test PR-AUC when a feature is shuffled.

### D23. Ablation study
- Rejected: using the model without checking what it depends on.
- Reason: if performance collapses without the time-window features, the model is learning how the simulator schedules laundering rather than how launderers behave. Variants tested: full model, without active span, without all time-window features, top 5 features only.

### D24. Seed stability and bootstrap confidence intervals
- Rejected: reporting a single run.
- Reason: the main claim has to hold across training seeds (3 seeds) and sampling noise (200 bootstrap resamples).
- Caveat: accounts in one laundering scheme are correlated, so the bootstrap intervals are somewhat narrower than the true ones.

### D25. Shared configuration module (scripts/common.py)
- Rejected: repeating features, model parameters and rules in each script.
- Reason: one definition avoids scripts drifting out of sync.

### D26. Final model drops active_span_hours
- Rejected: keeping the feature with the highest gain importance.
- Evidence:
  - PR-AUC improved without it, from 0.174 to 0.181 (seed standard deviation about 0.003). Removing all time-window features cost only 0.011 PR-AUC and made no difference to cases caught at the R3 budget (858 against 853), so the model does not depend on simulation timing.
  - Gain and permutation importance disagree: `active_span_hours` has 22.4% of gain but ranks ninth by permutation, while `max_txns_out_per_hour` has 1.8% of gain but ranks second.
  - A model on the top 5 features alone scores PR-AUC 0.064 against 0.174, so the value comes from combining features.
- Reason: a simpler model that scores better and depends less on timing is preferable.
- Result on 15 features: 3.0x more laundering accounts caught than the best rule at the same workload, bootstrap 95% CI 2.73x to 3.41x, stable across seeds. PR-AUC 0.174 (95% CI 0.160 to 0.191).
- Result on the final 14 features: PR-AUC 0.183. At the R3 budget (6,348 alerts), 891 against 285 caught (3.1x), precision 14.0% against 4.5%, cost per case found $142 against $445 (68% lower), recall on undocumented schemes 16.7% against 3.7%.

## Tableau

### D27. Pre-aggregated CSV exports
- Rejected: loading the raw fact table into Tableau (slow, close to platform limits), a live DuckDB connection (not supported by Tableau Public), business logic in Tableau calculated fields (hard to test and version).
- Reason: SQL scripts under version control do the calculations and Tableau only displays the results. CSV is the most reliable connector in Tableau Public. The largest export is about 373K account rows.
- A data dictionary (docs/tableau_data_dictionary.md) is generated with each file's grain, row count, columns and purpose.

### D28. One evaluation period for the what-if dashboard
- Rejected: mixing full-window rule results with test-period model results in one comparison.
- Reason: the alert budget slider must compare like with like, so rules and model are both applied to the same Sep 6-10 accounts (account_scores.csv). Full-window rule tables are used only for context.
- Note: Tableau Public workbooks are public. This is acceptable because the data is synthetic.

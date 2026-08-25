-- ============================================================================
-- Query 08: Risk-decile prioritization list, using window functions
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION:
-- "If care management can only act on the riskiest 10% of discharges, how
--  much of the total readmission burden would that 10% actually capture?"
--
-- WHY IT MATTERS:
-- This is a "gains/lift" style analysis -- exactly how the Phase 6 ML models
-- will ultimately be judged (per the Phase 1 success metric: precision at
-- top-K% risk). Building it here, in SQL, with a simple RULE-BASED risk
-- score (NOT a trained model -- just weighted, domain-justified EDA
-- findings), gives us a concrete number the Phase 6 model has to beat to
-- prove it's worth the added complexity.
--
-- risk_score weighting rationale (all weights from EDA-confirmed effect
-- sizes, not arbitrary):
--   service_utilization   x3  -- strongest standalone signal (Phase 3.5)
--   had_prior_inpatient    x2  -- ~2x risk multiplier on its own (Phase 4.9)
--   num_med_increased      x2  -- proxy for poor diabetes control (Phase 4.8)
--   number_diagnoses       x1  -- encounter complexity, weaker signal
--
-- NTILE(10) is a SQL window function that splits ranked rows into 10 equal-
-- sized buckets (deciles) -- it's doing the same job a data scientist would
-- do in pandas with pd.qcut, directly in the database.
-- ============================================================================

WITH scored AS (
    SELECT
        encounter_id,
        readmitted_30d,
        (service_utilization * 3 + had_prior_inpatient * 2
         + num_med_increased * 2 + number_diagnoses)      AS risk_score
    FROM encounters
),
deciled AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY risk_score DESC)          AS risk_decile   -- 1 = highest risk
    FROM scored
)
SELECT
    risk_decile,
    COUNT(*)                                               AS n_encounters,
    SUM(readmitted_30d)                                    AS n_readmissions,
    ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2)        AS readmit_rate_pct,
    ROUND(100.0 * SUM(readmitted_30d)
          / (SELECT SUM(readmitted_30d) FROM encounters), 2) AS pct_of_all_readmissions_captured
FROM deciled
GROUP BY risk_decile
ORDER BY risk_decile;

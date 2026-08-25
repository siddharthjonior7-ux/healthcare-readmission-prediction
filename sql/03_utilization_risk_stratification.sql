-- ============================================================================
-- Query 03: Risk stratification by prior healthcare utilization
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION:
-- "If we had to build a simple, explainable triage rule TODAY -- without
--  waiting for the ML model in Phase 6 -- how would readmission risk look
--  across utilization tiers, and how many patients fall in each tier?"
--
-- WHY IT MATTERS:
-- This directly operationalizes the strongest EDA finding (Phase 3.5): prior
-- utilization dominates other signals. CASE WHEN buckets turn a continuous
-- feature into hospital-operations language ("low/medium/high/very high
-- utilization") that a care coordinator can act on without touching a model.
-- ============================================================================

SELECT
    CASE
        WHEN service_utilization = 0 THEN '0 - No prior visits'
        WHEN service_utilization BETWEEN 1 AND 2 THEN '1-2 - Low utilization'
        WHEN service_utilization BETWEEN 3 AND 5 THEN '3-5 - Moderate utilization'
        ELSE '6+ - High utilization'
    END                                                  AS utilization_tier,
    COUNT(*)                                             AS n_encounters,
    SUM(readmitted_30d)                                  AS n_readmissions,
    ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2)      AS readmit_rate_pct
FROM encounters
GROUP BY utilization_tier
ORDER BY MIN(service_utilization);

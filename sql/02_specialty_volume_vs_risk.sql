-- ============================================================================
-- Query 02: Medical specialty -- volume vs. readmission risk
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION:
-- "Which admitting specialties see the highest readmission RATES, and among
--  those, which also have enough VOLUME to be worth a targeted intervention
--  program?"
--
-- WHY IT MATTERS:
-- A specialty with a 25% readmission rate on 40 patients is statistical
-- noise; a specialty with a 18% rate on 4,000 patients is a genuine
-- improvement opportunity. The HAVING clause enforces a minimum volume
-- threshold so the ranking isn't dominated by small, noisy groups --
-- exactly the kind of "n alongside the rate" discipline flagged in EDA.
-- ============================================================================

SELECT
    medical_specialty,
    COUNT(*)                                          AS n_encounters,
    ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2)   AS readmit_rate_pct
FROM encounters
GROUP BY medical_specialty
HAVING COUNT(*) >= 500          -- minimum volume threshold to avoid noisy small groups
ORDER BY readmit_rate_pct DESC;

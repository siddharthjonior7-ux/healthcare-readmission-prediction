-- ============================================================================
-- Query 01: Overall readmission rate + breakdown by primary diagnosis group
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION:
-- "What is our baseline 30-day readmission rate, and which primary diagnosis
--  categories are driving it? Where should clinical leadership focus first?"
--
-- WHY IT MATTERS:
-- Before targeting any intervention, a care management team needs a baseline
-- and a ranked list of which clinical conditions contribute most to
-- readmissions -- both in RATE (risk per patient) and VOLUME (total patients
-- affected), since a high-rate/low-volume group is a very different problem
-- than a high-volume/moderate-rate group.
-- ============================================================================

-- Overall baseline
SELECT
    COUNT(*)                                            AS total_encounters,
    SUM(readmitted_30d)                                 AS total_readmissions,
    ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2)     AS readmit_rate_pct
FROM encounters;

-- Breakdown by primary diagnosis group
SELECT
    diag_1_group,
    COUNT(*)                                            AS n_encounters,
    SUM(readmitted_30d)                                 AS n_readmissions,
    ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2)     AS readmit_rate_pct,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM encounters), 2) AS pct_of_all_encounters
FROM encounters
GROUP BY diag_1_group
ORDER BY readmit_rate_pct DESC;

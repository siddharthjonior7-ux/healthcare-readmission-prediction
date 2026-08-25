-- ============================================================================
-- Query 04: Length of stay vs. readmission risk
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION:
-- "Do longer hospital stays predict higher or lower readmission risk? This
--  has a direct operational tension: shorter stays reduce cost per encounter,
--  but might increase readmission risk if patients are discharged before
--  they're ready."
--
-- WHY IT MATTERS:
-- This is a classic hospital operations trade-off question -- length of stay
-- (LOS) reduction is a common cost-cutting target, but only makes sense if it
-- doesn't push readmissions up and erase the savings via HRRP penalties.
-- ============================================================================

SELECT
    time_in_hospital                                     AS length_of_stay_days,
    COUNT(*)                                              AS n_encounters,
    ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2)       AS readmit_rate_pct,
    ROUND(AVG(num_medications), 1)                         AS avg_medications,
    ROUND(AVG(num_procedures), 1)                           AS avg_procedures
FROM encounters
GROUP BY time_in_hospital
ORDER BY length_of_stay_days;

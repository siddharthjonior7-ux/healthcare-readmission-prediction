-- ============================================================================
-- Query 05: Readmission risk by admission type and source (with JOINs)
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION:
-- "Does HOW a patient enters the hospital (emergency vs. elective vs.
--  physician referral, etc.) relate to their readmission risk?"
--
-- WHY IT MATTERS:
-- Admission type/source are stored as numeric IDs in the main table --
-- exactly the kind of normalized schema you'd find in a real hospital data
-- warehouse. JOINing to the lookup tables (built from IDs_mapping.csv in
-- Phase 2/5) turns opaque codes into a report a hospital administrator can
-- actually read, without duplicating text data across 99,320 rows.
-- ============================================================================

SELECT
    lt.description                                        AS admission_type,
    COUNT(*)                                               AS n_encounters,
    ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(*), 2)      AS readmit_rate_pct
FROM encounters e
JOIN lookup_admission_type lt
    ON e.admission_type_id = lt.admission_type_id
GROUP BY lt.description
HAVING COUNT(*) >= 200
ORDER BY readmit_rate_pct DESC;

-- Same pattern for admission SOURCE
SELECT
    ls.description                                        AS admission_source,
    COUNT(*)                                               AS n_encounters,
    ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(*), 2)      AS readmit_rate_pct
FROM encounters e
JOIN lookup_admission_source ls
    ON e.admission_source_id = ls.admission_source_id
GROUP BY ls.description
HAVING COUNT(*) >= 200
ORDER BY readmit_rate_pct DESC;

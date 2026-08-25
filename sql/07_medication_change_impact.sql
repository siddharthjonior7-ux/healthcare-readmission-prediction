-- ============================================================================
-- Query 07: Impact of diabetes medication changes on readmission
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION:
-- "Does adjusting a patient's diabetes medication during their stay relate
--  to their readmission risk -- and does insulin use specifically matter?"
--
-- WHY IT MATTERS:
-- A dosage change often signals the care team found the patient's diabetes
-- poorly controlled (Phase 2.2.F). If medication-change patients are
-- consistently higher risk, that's a discharge-planning signal: these
-- patients may need closer follow-up on their NEW regimen, not just their
-- diagnosis.
-- ============================================================================

SELECT
    CASE WHEN change = 1 THEN 'Medication changed' ELSE 'No change' END AS med_change_status,
    COUNT(*)                                              AS n_encounters,
    ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2)       AS readmit_rate_pct
FROM encounters
GROUP BY med_change_status;

-- Insulin specifically (No / Steady / Down / Up, ordinal-encoded 0-3 in Phase 4)
SELECT
    CASE insulin
        WHEN 0 THEN 'No'
        WHEN 1 THEN 'Down'
        WHEN 2 THEN 'Steady'
        WHEN 3 THEN 'Up'
    END                                                    AS insulin_status,
    COUNT(*)                                               AS n_encounters,
    ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2)        AS readmit_rate_pct
FROM encounters
GROUP BY insulin
ORDER BY insulin;

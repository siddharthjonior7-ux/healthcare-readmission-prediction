-- ============================================================================
-- Query 06: Highest-risk patient segments (diagnosis x utilization), via CTE
-- ----------------------------------------------------------------------------
-- BUSINESS QUESTION:
-- "If we could only design ONE targeted care-management program, which
--  specific combination of diagnosis category + prior utilization tier
--  should it target, to get the best mix of high risk AND meaningful volume?"
--
-- WHY IT MATTERS:
-- Single-dimension analysis (Query 01, 03) can miss interaction effects.
-- A patient with a Circulatory diagnosis AND high prior utilization may be
-- meaningfully riskier than either factor alone would suggest. The CTE
-- (WITH clause) keeps the segment-building logic and the final filter/rank
-- step readable as two separate concerns, the way you'd structure this in
-- a real analytics codebase rather than one dense nested query.
-- ============================================================================

WITH segments AS (
    SELECT
        diag_1_group,
        CASE
            WHEN service_utilization = 0 THEN 'No prior visits'
            WHEN service_utilization BETWEEN 1 AND 2 THEN 'Low utilization'
            WHEN service_utilization BETWEEN 3 AND 5 THEN 'Moderate utilization'
            ELSE 'High utilization'
        END AS utilization_tier,
        readmitted_30d
    FROM encounters
)
SELECT
    diag_1_group,
    utilization_tier,
    COUNT(*)                                              AS n_encounters,
    ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2)       AS readmit_rate_pct
FROM segments
GROUP BY diag_1_group, utilization_tier
HAVING COUNT(*) >= 300                -- minimum segment size to act on operationally
ORDER BY readmit_rate_pct DESC
LIMIT 10;

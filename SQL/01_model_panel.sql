-- =====================================================================
-- Machine learning panel, built on the hospital_finance database from the companion analytics project
-- (US_Hospital_Financial_Performance_Analysis). One row per hospital per year t.
--
-- Target (distress): the hospital lost money in BOTH of the next two years (t+1 and t+2).
--   Only rows where both later reports exist, for consecutive years, get a label. Hospitals that close have
--   no later reports, so closures are not in the label (see README, limitations).
-- Features: only what is known at the end of year t (this year, the two years before, and the state and
--   national picture that year). Nothing from t+1 or later.
-- =====================================================================

DROP SCHEMA IF EXISTS ml CASCADE;
CREATE SCHEMA ml;

CREATE VIEW ml.v_model_panel AS
WITH h AS (           -- study hospital-years with the usual cleaning (no partial years, no impossible margins)
    SELECT v.*,
           lag(fiscal_year, 1)      OVER w AS y_1,
           lag(fiscal_year, 2)      OVER w AS y_2,
           lag(total_margin, 1)     OVER w AS margin_1,
           lag(total_margin, 2)     OVER w AS margin_2,
           lag(total_revenue, 1)    OVER w AS revenue_1,
           lag(discharges, 1)       OVER w AS discharges_1,
           lag(cost_per_discharge, 1) OVER w AS cost_per_discharge_1,
           lag(contract_labor_pct, 1) OVER w AS contract_labor_1,
           lag(operating_margin, 1) OVER w AS operating_margin_1,
           lag(other_income / nullif(total_revenue, 0), 1) OVER w AS other_income_share_1,
           lag(days_cash_on_hand, 1) OVER w AS days_cash_1,
           lag(current_ratio, 1)    OVER w AS current_ratio_1,
           (net_income < 0)::int AS loss,
           -- gaps-and-islands: years in a row with a loss, up to and including this year
           fiscal_year - row_number() OVER (PARTITION BY ccn, net_income < 0 ORDER BY fiscal_year) AS grp
    FROM analytics.v_hospital_year v
    WINDOW w AS (PARTITION BY ccn ORDER BY fiscal_year)
), streak AS (
    SELECT ccn, fiscal_year,
           CASE WHEN loss = 1 THEN count(*) OVER (PARTITION BY ccn, loss, grp ORDER BY fiscal_year) ELSE 0 END AS loss_streak
    FROM h
), future AS (        -- the label: losses in the next two years, from every report (outliers included)
    SELECT ccn, fiscal_year,
           lead(fiscal_year, 1) OVER w = fiscal_year + 1 AND lead(fiscal_year, 2) OVER w = fiscal_year + 2 AS has_future,
           (lead(net_income, 1) OVER w < 0 AND lead(net_income, 2) OVER w < 0)::int                    AS distress_next_2y,
           (lead(net_income, 1) OVER w < 0)::int                                                       AS loss_next_year
    FROM core.fact_hospital_year
    WINDOW w AS (PARTITION BY ccn ORDER BY fiscal_year)
), state_year AS (
    SELECT state_abbrev, fiscal_year,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY total_margin) AS state_median_margin,
           avg(loss)                                                  AS state_share_losing
    FROM h GROUP BY 1, 2
), nation_year AS (
    SELECT fiscal_year, percentile_cont(0.5) WITHIN GROUP (ORDER BY total_margin) AS us_median_margin
    FROM h GROUP BY 1
)
SELECT h.ccn, h.fiscal_year, h.hospital_name, h.city, h.state_abbrev,
       h.hospital_type, h.ownership, coalesce(h.rural_urban, 'Unknown') AS rural_urban, h.medicaid_status,
       h.expanded_now::int AS medicaid_expanded_now,
       -- profitability and its history
       h.total_margin, h.operating_margin,
       CASE WHEN h.y_1 = h.fiscal_year - 1 THEN h.margin_1 END                                      AS margin_last_year,
       CASE WHEN h.y_2 = h.fiscal_year - 2 THEN h.margin_2 END                                      AS margin_two_years_ago,
       CASE WHEN h.y_1 = h.fiscal_year - 1 THEN h.total_margin - h.margin_1 END                     AS margin_change,
       CASE WHEN h.y_1 = h.fiscal_year - 1 THEN h.operating_margin_1 END                            AS operating_margin_last_year,
       h.loss, s.loss_streak,
       -- how much of the income is NOT from patient care (investments, donations, grants, COVID relief):
       -- it swings with the stock market and can disappear the next year
       h.other_income / nullif(h.total_revenue, 0)                                                  AS other_income_share,
       CASE WHEN h.y_1 = h.fiscal_year - 1 THEN h.other_income_share_1 END                          AS other_income_share_last_year,
       -- size and growth
       h.beds, ln(nullif(h.total_revenue, 0))                                                       AS log_revenue,
       CASE WHEN h.y_1 = h.fiscal_year - 1 THEN h.total_revenue / nullif(h.revenue_1, 0) - 1 END    AS revenue_growth,
       CASE WHEN h.y_1 = h.fiscal_year - 1 THEN h.discharges / nullif(h.discharges_1, 0) - 1 END    AS discharge_growth,
       h.occupancy_rate,
       -- balance sheet (cash is unreliable for system hospitals; kept as a feature, not as the label)
       h.current_ratio, h.debt_ratio, h.days_cash_on_hand,
       CASE WHEN h.y_1 = h.fiscal_year - 1 THEN h.days_cash_on_hand - h.days_cash_1 END             AS days_cash_change,
       CASE WHEN h.y_1 = h.fiscal_year - 1 THEN h.current_ratio - h.current_ratio_1 END             AS current_ratio_change,
       h.salaries / nullif(h.total_costs, 0)                                                        AS salary_share_of_costs,
       h.fte_employees / nullif(h.beds, 0)                                                          AS staff_per_bed,
       -- costs, prices and staffing
       h.cost_per_discharge,
       CASE WHEN h.y_1 = h.fiscal_year - 1 THEN h.cost_per_discharge / nullif(h.cost_per_discharge_1, 0) - 1 END
                                                                                                    AS cost_growth,
       h.charge_to_cost, h.contract_labor_pct,
       -- patient mix
       h.medicaid_day_share, h.medicare_day_share, h.uncompensated_care_pct,
       -- the environment that year
       sy.state_median_margin, sy.state_share_losing, ny.us_median_margin,
       -- labels
       f.has_future, f.distress_next_2y, f.loss_next_year
FROM h
JOIN streak s USING (ccn, fiscal_year)
JOIN future f USING (ccn, fiscal_year)
JOIN state_year sy USING (state_abbrev, fiscal_year)
JOIN nation_year ny USING (fiscal_year);

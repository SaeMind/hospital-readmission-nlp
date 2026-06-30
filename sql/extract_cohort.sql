-- sql/extract_cohort.sql
-- Extract model-ready cohort from normalized SQLite schema.
-- Joins patients + admissions + discharge_notes back into the flat
-- format expected by MIMICDischargeDataset (subject_id, hadm_id,
-- discharge_summary, readmitted_30d).
--
-- Usage (sqlite3 CLI):
--   sqlite3 data/readmission.db < sql/extract_cohort.sql
--
-- Usage (Python):
--   pd.read_sql_query(open("sql/extract_cohort.sql").read(), conn)

SELECT
    p.subject_id,
    a.hadm_id,
    n.discharge_summary,
    a.readmitted_30d
FROM patients        AS p
JOIN admissions      AS a ON a.subject_id = p.subject_id
JOIN discharge_notes AS n ON n.hadm_id    = a.hadm_id
ORDER BY a.hadm_id;

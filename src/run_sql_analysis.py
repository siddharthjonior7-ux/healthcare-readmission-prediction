"""
run_sql_analysis.py
--------------------
Phase 5: Executes every .sql file in sql/ against the cleaned dataset
(loaded into SQLite for portability -- the same queries run unmodified on
PostgreSQL/MySQL with only trivial syntax differences, since they use
standard ANSI SQL: CTEs, window functions, JOINs, GROUP BY/HAVING).

Run as: python src/run_sql_analysis.py   (from project root)
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/processed/readmission.db")
SQL_DIR = Path("sql")
OUT_PATH = Path("outputs/sql_results.txt")


def build_database():
    """(Re)build the SQLite database from the cleaned CSV + lookup tables."""
    conn = sqlite3.connect(DB_PATH)
    encounters = pd.read_csv("data/processed/cleaned_diabetic_data.csv")
    encounters.to_sql("encounters", conn, if_exists="replace", index=False)
    for name in ["admission_type", "discharge_disposition", "admission_source"]:
        # na_filter=False: prevents pandas from silently converting the
        # literal text "NULL" (a real category in this codebook, e.g.
        # admission_type_id 6) into an actual missing value on reload.
        df = pd.read_csv(f"data/raw/lookup_{name}.csv", na_filter=False)
        df.to_sql(f"lookup_{name}", conn, if_exists="replace", index=False)
    conn.close()


def run_all_queries():
    conn = sqlite3.connect(DB_PATH)
    lines = []
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        lines.append(f"\n{'='*80}\n{sql_file.name}\n{'='*80}")
        print(f"\n{'='*80}\n{sql_file.name}\n{'='*80}")

        raw = sql_file.read_text()
        # Strip full-line comments FIRST, then split on semicolons -- doing
        # it in the other order breaks if a comment sentence happens to
        # contain a semicolon (as one does in this file).
        code_only = "\n".join(
            l for l in raw.split("\n") if not l.strip().startswith("--")
        )
        statements = [s.strip() for s in code_only.split(";") if s.strip()]
        for stmt in statements:
            try:
                result = pd.read_sql_query(stmt, conn)
                lines.append(result.to_string(index=False))
                print(result.to_string(index=False))
            except Exception as e:
                lines.append(f"[ERROR running statement: {e}]")
                print(f"[ERROR running statement: {e}]")
    conn.close()
    OUT_PATH.write_text("\n".join(lines))
    print(f"\nAll query results saved to {OUT_PATH.resolve()}")


if __name__ == "__main__":
    build_database()
    run_all_queries()

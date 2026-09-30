"""Build the modelling panel in PostgreSQL (SQL/01_model_panel.sql) and export it to Data/model_panel.csv.gz.

    python Python/01_build_dataset.py

Needs the hospital_finance database built by the companion analytics project
(US_Hospital_Financial_Performance_Analysis: python run_all.py). Other scripts read the CSV through load_panel(),
so the model, notebook and app run without a database.
"""
import os
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "SQL" / "01_model_panel.sql"
OUT = ROOT / "Data" / "model_panel.csv.gz"
CONNINFO = (f"host={os.getenv('PGHOST', 'localhost')} port={os.getenv('PGPORT', '5432')} "
            f"dbname={os.getenv('PGDATABASE', 'hospital_finance')} user={os.getenv('PGUSER', 'postgres')}")

CATEGORICAL = ["hospital_type", "ownership", "rural_urban", "medicaid_status"]


def load_panel():
    """The exported panel with the right types (used by every other script)."""
    df = pd.read_csv(OUT, dtype={"ccn": str})
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    # a label needs both later reports AND a reported net income in them
    df["has_future"] = df.has_future.astype(bool) & df.distress_next_2y.notna()
    return df


def main():
    with psycopg.connect(CONNINFO) as conn:
        conn.execute(SQL.read_text(encoding="utf-8-sig"))
        cur = conn.execute("SELECT * FROM ml.v_model_panel ORDER BY ccn, fiscal_year")
        df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: v.__class__.__name__ == "Decimal").any():
            df[c] = df[c].astype(float)
    df["has_future"] = df.has_future.fillna(False).astype(bool)
    df.loc[~df.has_future, ["distress_next_2y", "loss_next_year"]] = pd.NA
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    lab = df[df.has_future]
    print(f"{len(df):,} hospital-years, {df.ccn.nunique():,} hospitals, {df.fiscal_year.min()}-{df.fiscal_year.max()}")
    print(f"labelled rows: {len(lab):,}; share in distress next 2 years: {lab.distress_next_2y.mean():.1%}")
    print(lab.groupby("fiscal_year").distress_next_2y.agg(["size", "mean"]).round(3).to_string())
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

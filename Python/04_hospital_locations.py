"""Map position for every scored hospital, for the app's night map.

    python Python/04_hospital_locations.py

Takes each hospital's ZIP code from its latest cost report (hospital_finance database, raw layer) and places it at
the centre of that ZIP code (Census Bureau 2023 ZCTA gazetteer). A tiny deterministic jitter keeps hospitals that
share a ZIP code from sitting exactly on top of each other. Writes app/data/hospital_locations.csv.
"""
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "raw"
URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip"
ZIPFILE = RAW / "2023_Gaz_zcta_national.zip"
OUT = ROOT / "app" / "data" / "hospital_locations.csv"
ANALYTICS_RAW = Path(os.getenv("ANALYTICS_RAW", ROOT.parent / "US_Hospital_Financial_Performance_Analysis" / "Data" / "raw"))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if not ZIPFILE.exists():
        subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-o", str(ZIPFILE), URL], check=True)
    with zipfile.ZipFile(ZIPFILE) as z:
        name = next(n for n in z.namelist() if n.endswith(".txt"))
        gaz = pd.read_csv(z.open(name), sep="\t", dtype={"GEOID": str})
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz.rename(columns={"GEOID": "zip", "INTPTLAT": "lat", "INTPTLONG": "lon"})[["zip", "lat", "lon"]]

    # ZIP codes come from the cost report files downloaded by the analytics project (the raw staging tables in
    # PostgreSQL are UNLOGGED, so they are empty after a server restart; the files are the durable copy)
    parts = []
    for f in sorted(ANALYTICS_RAW.glob("cost_report_*.csv")):
        d = pd.read_csv(f, usecols=["Provider CCN", "Zip Code"], dtype=str)
        parts.append(d.assign(year=int(f.stem[-4:])))
    z = (pd.concat(parts).rename(columns={"Provider CCN": "ccn", "Zip Code": "zip"}).dropna()
         .assign(zip=lambda d: d.zip.str[:5]).query("zip.str.match('^[0-9]{5}$')", engine="python")
         .sort_values("year").groupby("ccn").tail(1)[["ccn", "zip"]])
    scores = pd.read_csv(ROOT / "app" / "data" / "hospital_scores.csv", dtype={"ccn": str}, usecols=["ccn"])
    loc = scores.merge(z, on="ccn", how="left").merge(gaz, on="zip", how="left")
    # PO-box ZIP codes have no area of their own: use the nearest ZIP code by number (almost always the same town)
    miss = loc.lat.isna() & loc.zip.notna()
    zn = gaz.zip.astype(int).to_numpy()
    idx = [int(np.abs(zn - int(v)).argmin()) for v in loc.loc[miss, "zip"]]
    loc.loc[miss, ["lat", "lon"]] = gaz.iloc[idx][["lat", "lon"]].to_numpy()
    # jitter of up to ~2 km, seeded by the hospital's CCN so the map never changes between builds
    rng = np.random.default_rng(loc.ccn.map(lambda c: int("".join(ch for ch in c if ch.isdigit()) or 0)).sum() % 2**32)
    loc["lat"] += rng.uniform(-0.02, 0.02, len(loc))
    loc["lon"] += rng.uniform(-0.02, 0.02, len(loc))
    loc[["ccn", "lat", "lon"]].round(4).to_csv(OUT, index=False)
    print(f"placed {loc.lat.notna().sum():,} of {len(loc):,} hospitals ({loc.lat.notna().mean():.1%}); wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

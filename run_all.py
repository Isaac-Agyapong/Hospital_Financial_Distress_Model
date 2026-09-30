"""Rebuild the machine learning project in order.

    python run_all.py

Needs the hospital_finance PostgreSQL database from the companion analytics project
(US_Hospital_Financial_Performance_Analysis: python run_all.py). Steps: build the model panel -> tune, test and train
the model (writes models/ and app/data/) -> build and run the report notebook. Then: streamlit run app/app.py
"""
import subprocess
import sys
import time
from pathlib import Path

PY = Path(__file__).resolve().parent / "Python"
STEPS = [("01_build_dataset.py", "build the model panel in PostgreSQL and export Data/model_panel.csv.gz"),
         ("02_train_model.py", "tune, backtest, train the final model, score every hospital"),
         ("03_build_notebook.py", "build and execute Python/03_model_report.ipynb")]

if __name__ == "__main__":
    start = time.time()
    for script, what in STEPS:
        print(f"\n== {script}: {what}")
        subprocess.run([sys.executable, script], cwd=PY, check=True)
    print(f"\nall steps finished in {(time.time() - start) / 60:.1f} min")

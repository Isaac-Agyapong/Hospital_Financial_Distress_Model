"""Train and test the hospital financial distress model.

    python Python/02_train_model.py

Question: from a hospital's report for year t, will it lose money in BOTH of the next two years (t+1, t+2)?

Time-based evaluation (a hospital's future is never used to predict its own past):
    tune       train on years <= 2016 (labels end 2018), validate on 2018
    backtests  train on years <= t-2, test on t, for t = 2019, 2020, 2021 (labels end t+2 <= 2023)
               the main test is 2021: predicting 2022-2023, the two years after COVID relief ended
    final      train on every labelled year (<= 2021), score each hospital's latest report (2023 -> risk for 2024-2025)

Compared with rules a manager could use without a model: "lost money this year", "already lost money two
years running", and "lowest profit margin first". Writes models/*.json|csv and app/data/hospital_scores.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Python"))
from importlib import import_module  # noqa: E402

build = import_module("01_build_dataset")
MODELS, APP_DATA = ROOT / "models", ROOT / "app" / "data"
MODELS.mkdir(exist_ok=True)
APP_DATA.mkdir(parents=True, exist_ok=True)

TARGET = "distress_next_2y"
NUMERIC = ["total_margin", "operating_margin", "margin_last_year", "margin_two_years_ago", "margin_change",
           "operating_margin_last_year", "loss", "loss_streak", "other_income_share", "other_income_share_last_year",
           "beds", "log_revenue", "revenue_growth", "discharge_growth", "occupancy_rate",
           "current_ratio", "debt_ratio", "days_cash_on_hand", "days_cash_change", "current_ratio_change",
           "salary_share_of_costs", "staff_per_bed", "cost_per_discharge", "cost_growth", "charge_to_cost",
           "contract_labor_pct", "medicaid_day_share", "medicare_day_share", "uncompensated_care_pct",
           "medicaid_expanded_now", "state_median_margin", "state_share_losing", "us_median_margin"]
CATEGORICAL = build.CATEGORICAL
FEATURES = NUMERIC + CATEGORICAL
TOP_SHARE = 0.10            # the watch list: the 10% of hospitals with the highest risk
SEED = 7

# plain-language names for the reasons shown in the app and README
LABELS = {
    "total_margin": "profit margin this year", "operating_margin": "profit on patient care",
    "margin_last_year": "profit margin last year", "margin_two_years_ago": "profit margin two years ago",
    "margin_change": "change in profit margin from last year", "loss": "lost money this year",
    "operating_margin_last_year": "profit on patient care last year",
    "other_income_share": "income from investments, gifts and grants",
    "other_income_share_last_year": "income from investments, gifts and grants last year",
    "days_cash_change": "change in cash in the bank", "current_ratio_change": "change in ability to pay short-term bills",
    "salary_share_of_costs": "staff pay as a share of costs", "staff_per_bed": "staff per bed",
    "loss_streak": "years in a row losing money", "beds": "number of beds", "log_revenue": "size (revenue)",
    "revenue_growth": "revenue growth", "discharge_growth": "change in patients treated",
    "occupancy_rate": "share of beds filled", "current_ratio": "short-term bills it can cover",
    "debt_ratio": "debt compared with assets", "days_cash_on_hand": "days of cash in the bank",
    "cost_per_discharge": "cost per patient stay", "cost_growth": "growth in cost per stay",
    "charge_to_cost": "list prices compared with costs", "contract_labor_pct": "spending on agency staff",
    "medicaid_day_share": "share of Medicaid patients", "medicare_day_share": "share of Medicare patients",
    "uncompensated_care_pct": "unpaid care", "medicaid_expanded_now": "state has expanded Medicaid",
    "state_median_margin": "typical margin in its state", "state_share_losing": "share of hospitals losing money in its state",
    "us_median_margin": "national typical margin", "hospital_type": "hospital type", "ownership": "owner",
    "rural_urban": "rural or urban", "medicaid_status": "state Medicaid expansion",
}


def xgb_model(**kw):
    params = dict(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=10, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=2.0, enable_categorical=True, tree_method="hist",
                  eval_metric="aucpr", random_state=SEED, n_jobs=-1)
    params.update(kw)
    return xgb.XGBClassifier(**params)


def logit_model():
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))


def clip(df):
    """Winsorise extreme ratios (a few reports have nonsense values) using limits learned on the training rows."""
    return df


def top_k_stats(y, score, share=TOP_SHARE):
    k = int(round(len(y) * share))
    order = np.argsort(-np.asarray(score), kind="stable")[:k]
    hits = int(np.asarray(y)[order].sum())
    return {"flagged": k, "hits": hits, "precision_top": hits / k, "recall_top": hits / int(np.sum(y))}


def evaluate(y, score, name):
    return {"model": name, "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
            **top_k_stats(y, score)}


def baselines(df):
    """Rules of thumb. Ties are broken by profit margin so every rule can rank all hospitals.
    ("Lost money this year" ranks exactly like "lowest margin first", because a loss is a margin below zero.)"""
    tiebreak = -df.total_margin.fillna(0) / 100
    return {"Rule: lost money 2+ years in a row": (df.loss_streak >= 2).astype(int) + tiebreak,
            "Rule: lowest profit margin first": -df.total_margin.fillna(df.total_margin.median())}


def winsor_limits(train):
    cols = [c for c in NUMERIC if c not in ("loss", "loss_streak", "medicaid_expanded_now", "beds")]
    return {c: (train[c].quantile(0.005), train[c].quantile(0.995)) for c in cols}


def apply_limits(df, limits):
    df = df.copy()
    for c, (lo, hi) in limits.items():
        df[c] = df[c].clip(lo, hi)
    return df


def tune(panel):
    """Small grid, scored on two validation years that are never used for testing (2017 and 2018, each
    predicted by a model trained only on years whose outcomes were known by then). Average PR-AUC decides."""
    lab = panel[panel.has_future]
    folds = []
    for v in (2017, 2018):
        tr, va = lab[lab.fiscal_year <= v - 2], lab[lab.fiscal_year == v]
        lim = winsor_limits(tr)
        folds.append((apply_limits(tr, lim), apply_limits(va, lim)))
    best = None
    for depth in (3, 4, 5):
        for mcw in (10, 30):
            for n in (300, 600, 1000):
                params = dict(max_depth=depth, min_child_weight=mcw, learning_rate=0.03, n_estimators=n)
                aps = []
                for tr, va in folds:
                    m = xgb_model(**params).fit(tr[FEATURES], tr[TARGET].astype(int))
                    aps.append(average_precision_score(va[TARGET].astype(int), m.predict_proba(va[FEATURES])[:, 1]))
                if best is None or np.mean(aps) > best[0]:
                    best = (float(np.mean(aps)), params)
    print(f"tuning: best mean validation PR-AUC {best[0]:.3f} with {best[1]}")
    return best[1]


def backtest(panel, params):
    lab = panel[panel.has_future]
    rows, test_preds = [], []
    for t in (2019, 2020, 2021):
        tr, te = lab[lab.fiscal_year <= t - 2], lab[lab.fiscal_year == t]
        lim = winsor_limits(tr)
        tr, te = apply_limits(tr, lim), apply_limits(te, lim)
        y = te[TARGET].astype(int)
        m = xgb_model(**params).fit(tr[FEATURES], tr[TARGET].astype(int))
        lr = logit_model().fit(tr[FEATURES], tr[TARGET].astype(int))
        scores = {"XGBoost": m.predict_proba(te[FEATURES])[:, 1],
                  "Logistic regression": lr.predict_proba(te[FEATURES])[:, 1], **baselines(te)}
        for name, sc in scores.items():
            rows.append({"test_year": t, "outcome_years": f"{t + 1}-{t + 2}", "group": "All hospitals", "n": len(te),
                         "base_rate": float(y.mean()), **evaluate(y, sc, name)})
        # early warning: hospitals that MADE money in year t. Rules based on past losses cannot rank them; the
        # only simple rule left is "thinnest profit margin first".
        ok = (te.loss == 0).values
        for name in ("XGBoost", "Logistic regression", "Rule: lowest profit margin first"):
            rows.append({"test_year": t, "outcome_years": f"{t + 1}-{t + 2}", "group": "Made money this year",
                         "n": int(ok.sum()), "base_rate": float(y[ok].mean()),
                         **evaluate(y[ok], np.asarray(scores[name])[ok], name)})
        if t == 2021:
            p = scores["XGBoost"]
            test_preds = te[["ccn", "fiscal_year", "hospital_name", "state_abbrev", "hospital_type", "rural_urban",
                             "total_margin", "loss", "loss_streak"]].assign(risk=p, actual=y.values)
            calib = (pd.DataFrame({"risk": p, "actual": y.values})
                     .assign(bin=lambda d: pd.qcut(d.risk, 10, labels=False, duplicates="drop"))
                     .groupby("bin").agg(predicted=("risk", "mean"), actual=("actual", "mean"), n=("risk", "size")))
            brier = brier_score_loss(y, p)
    res = pd.DataFrame(rows)
    return res, test_preds, calib, brier


def reasons(model, X, top=3):
    """Top SHAP reasons per hospital, in plain words, with the direction (raises or lowers risk)."""
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    out = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(sv[i]))[:top]
        out.append([(FEATURES[j], float(sv[i, j])) for j in order])
    return sv, out


def main():
    panel = build.load_panel()
    params = tune(panel)
    res, test_preds, calib, brier = backtest(panel, params)
    pd.set_option("display.width", 160)
    print(res.pivot_table(index=["group", "model"], columns="test_year", values=["roc_auc", "precision_top"])
          .round(3).to_string())

    # final model on every labelled year; score the latest report of every hospital
    lab = panel[panel.has_future]
    lim = winsor_limits(lab)
    final = xgb_model(**params).fit(apply_limits(lab, lim)[FEATURES], lab[TARGET].astype(int))
    final.save_model(MODELS / "xgb_final.json")
    # the app re-scores edited values on its "What if?" page, so it needs the model and the same clipping limits
    final.save_model(APP_DATA / "xgb_final.json")
    (APP_DATA / "winsor_limits.json").write_text(json.dumps({k: [float(a), float(b)] for k, (a, b) in lim.items()}),
                                                 encoding="utf-8")
    panel.loc[:, ["ccn", "fiscal_year", "total_margin"]].round(4).to_csv(APP_DATA / "margin_history.csv", index=False)
    (APP_DATA / "features.json").write_text(json.dumps({
        "numeric": NUMERIC, "categorical": CATEGORICAL, "labels": LABELS,
        # the model codes categories by their position in these lists, so the app must rebuild them identically
        "categories": {c: [str(v) for v in panel[c].cat.categories] for c in CATEGORICAL}}), encoding="utf-8")
    latest = panel.sort_values("fiscal_year").groupby("ccn").tail(1)
    latest = latest[latest.fiscal_year >= 2022].copy()
    Xl = apply_limits(latest, lim)[FEATURES]
    latest["risk"] = final.predict_proba(Xl)[:, 1]
    sv, why = reasons(final, Xl)
    latest["risk_rank"] = latest.risk.rank(ascending=False, method="first").astype(int)
    cut = latest.risk.quantile(1 - TOP_SHARE)
    latest["risk_level"] = np.select([latest.risk >= cut, latest.risk >= latest.risk.quantile(0.70)],
                                     ["High", "Elevated"], "Lower")
    for k in range(3):
        latest[f"reason_{k + 1}"] = [LABELS[w[k][0]] for w in why]
        latest[f"reason_{k + 1}_up"] = [w[k][1] > 0 for w in why]
    latest.to_csv(APP_DATA / "hospital_scores.csv", index=False)

    # global importance (mean |SHAP|) on the latest reports
    imp = (pd.DataFrame({"feature": FEATURES, "importance": np.abs(sv).mean(axis=0)})
           .assign(label=lambda d: d.feature.map(LABELS)).sort_values("importance", ascending=False))
    imp.to_csv(MODELS / "feature_importance.csv", index=False)
    res.to_csv(MODELS / "backtest.csv", index=False)
    test_preds.to_csv(MODELS / "test_predictions_2021.csv", index=False)
    calib.to_csv(MODELS / "calibration_2021.csv")
    main_test = res[(res.test_year == 2021) & (res.group == "All hospitals")].set_index("model")
    early = res[(res.test_year == 2021) & (res.group == "Made money this year")].set_index("model")
    summary = {
        "target": "lost money in both of the next two years", "top_share": TOP_SHARE, "params": params,
        "features": FEATURES, "labelled_rows": int(len(lab)), "hospitals_scored": int(len(latest)),
        "scored_year_counts": latest.fiscal_year.value_counts().sort_index().to_dict(),
        "test_2021": main_test.drop(columns=["test_year", "outcome_years", "group"]).round(4).to_dict(orient="index"),
        "early_warning_2021": early.drop(columns=["test_year", "outcome_years", "group"]).round(4).to_dict(orient="index"),
        "brier_2021": round(brier, 4), "high_risk_cutoff": round(float(cut), 4),
        "high_risk_hospitals": int((latest.risk_level == "High").sum()),
    }
    (MODELS / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nscored {len(latest):,} hospitals (latest report); high risk: {summary['high_risk_hospitals']}")
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

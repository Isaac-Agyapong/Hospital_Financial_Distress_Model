"""Hospital Financial Distress Forecast: Streamlit app.

    streamlit run app/app.py

Reads app/data/ (written by Python/02_train_model.py): the latest risk score for every hospital, the trained XGBoost
model (used live for explanations and the what-if simulator) and each hospital's profit history; and models/ for the
test results. Design: product-style "bento" layout (lavender canvas, white cards, indigo for the model, coral for
risk), Plus Jakarta Sans, Material icons in a sidebar, filters that drive every chart, click-through from the map and
the table to a hospital profile.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

APP = Path(__file__).resolve().parent
DATA, MODELS = APP / "data", APP.parent / "models"
INDIGO, INDIGO_D, INDIGO_L, INDIGO_XL = "#3F37C9", "#2B2596", "#A5A1EA", "#ECEBFB"
CORAL, AMBER, TEAL = "#E4572E", "#E8962E", "#12A594"
INK, INK_2, INK_3, CANVAS, CARD, LINE = "#16162A", "#5C6076", "#9094A6", "#F4F4F9", "#FFFFFF", "#E7E7F0"
LEVEL_COLOUR = {"High": CORAL, "Elevated": AMBER, "Lower": TEAL}
LEVEL_BG = {"High": "#FCE7E1", "Elevated": "#FDF0DE", "Lower": "#DDF4F1"}
LEVELS = ["High", "Elevated", "Lower"]

st.set_page_config(page_title="Hospital Financial Distress Forecast", page_icon=":material/monitor_heart:",
                   layout="wide", initial_sidebar_state="expanded")


# ================================================================================================= data and model
@st.cache_data
def load():
    s = pd.read_csv(DATA / "hospital_scores.csv", dtype={"ccn": str})
    s["city"] = s.city.fillna("").str.title()
    s["name"] = s.hospital_name.str.strip()
    s["label"] = s.name + " (" + s.city + ", " + s.state_abbrev + ")"
    dup = s.label.duplicated(keep=False)
    s.loc[dup, "label"] += " #" + s.loc[dup, "ccn"]
    s["type_short"] = s.hospital_type.map({"General acute care": "General", "Critical access": "Critical access"})
    s["percentile"] = 100 * s.risk.rank(pct=True)
    hist = pd.read_csv(DATA / "margin_history.csv", dtype={"ccn": str})
    meta = json.loads((DATA / "features.json").read_text(encoding="utf-8"))
    limits = json.loads((DATA / "winsor_limits.json").read_text(encoding="utf-8"))
    return s.sort_values("risk_rank").reset_index(drop=True), hist, meta, limits


@st.cache_data
def test_results():
    """Numbers quoted in the app, read from the saved test results so they always match the model."""
    bt = pd.read_csv(MODELS / "backtest.csv")
    t = pd.read_csv(MODELS / "test_predictions_2021.csv")
    q70, q90 = t.risk.quantile(0.7), t.risk.quantile(0.9)
    lvl = np.select([t.risk >= q90, t.risk >= q70], ["High", "Elevated"], "Lower")
    track = {k: round(100 * t.actual[lvl == k].mean()) for k in LEVELS}
    return bt, track


@st.cache_resource
def booster():
    m = xgb.XGBClassifier()
    m.load_model(DATA / "xgb_final.json")
    return m


SCORES, HIST, META, LIMITS = load()
LOC = pd.read_csv(DATA / "hospital_locations.csv", dtype={"ccn": str})
BT, TRACK = test_results()
N = len(SCORES)
FEATURES = META["numeric"] + META["categorical"]
Q70, Q90 = SCORES.risk.quantile(0.70), SCORES.risk.quantile(0.90)
MAIN = BT[(BT.test_year == 2021) & (BT.group == "All hospitals")].set_index("model")
EARLY = BT[(BT.test_year == 2021) & (BT.group == "Made money this year")].set_index("model")
HIT = round(100 * MAIN.loc["XGBoost", "precision_top"])


def frame(rows):
    X = pd.DataFrame(rows)[FEATURES].copy()
    for c in META["numeric"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c, (lo, hi) in LIMITS.items():
        X[c] = X[c].clip(lo, hi)
    for c in META["categorical"]:
        X[c] = pd.Categorical(X[c].astype(str), categories=META["categories"][c])
    return X


def predict(row):
    return float(booster().predict_proba(frame([row]))[:, 1][0])


def contributions(row):
    """Exact per-feature contributions (log-odds) from the trees: positive = pushes risk up."""
    X = frame([row])
    c = booster().get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[0]
    return pd.Series(c[:-1], index=FEATURES)


def level_of(risk):
    return "High" if risk >= Q90 else "Elevated" if risk >= Q70 else "Lower"


def percentile_of(risk):
    return 100 * (SCORES.risk <= risk).mean()


PCT = {"total_margin", "operating_margin", "margin_last_year", "margin_two_years_ago", "operating_margin_last_year",
       "other_income_share", "other_income_share_last_year", "revenue_growth", "discharge_growth", "occupancy_rate",
       "debt_ratio", "cost_growth", "contract_labor_pct", "medicaid_day_share", "medicare_day_share",
       "uncompensated_care_pct", "salary_share_of_costs", "state_median_margin", "state_share_losing",
       "us_median_margin"}


def show_value(f, v):
    if isinstance(v, float) and np.isnan(v) or v is None:
        return "not reported"
    if f in PCT:
        return f"{100 * v:.1f}%"
    if f == "margin_change":
        return f"{100 * v:+.1f} points"
    if f in ("days_cash_on_hand",):
        return "not reliable" if v < 0 else f"{v:.0f} days"
    if f == "days_cash_change":
        return f"{v:+.0f} days"
    if f == "cost_per_discharge":
        return f"${v:,.0f}"
    if f == "log_revenue":
        r = np.exp(v)
        return f"${r / 1e9:.1f}B revenue" if r >= 1e9 else f"${r / 1e6:.0f}M revenue"
    if f in ("current_ratio", "charge_to_cost"):
        return f"{v:.1f}x"
    if f == "current_ratio_change":
        return f"{v:+.2f}"
    if f in ("loss", "medicaid_expanded_now"):
        return "yes" if v else "no"
    if f == "staff_per_bed":
        return f"{v:.1f}"
    if f in ("loss_streak", "beds"):
        return f"{v:.0f}"
    return str(v)


# ================================================================================================= style
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stApp p, .stApp div, .stApp span, .stApp label, .stApp input, .stApp button, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
/* keep Streamlit's icon font (the rule above would otherwise turn icons into words) */
.stApp [data-testid="stIconMaterial"], .stApp span[class*="material"], .stApp .material-symbols-rounded {{
    font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {CANVAS}; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"] {{ top: 6px; }}
.block-container {{ padding: 1.6rem 2.2rem 2rem 2.2rem; max-width: 1320px; }}
h1 {{ font-size: 1.9rem !important; font-weight: 800 !important; letter-spacing: -0.03em; color: {INK}; margin: 0 !important; padding: 0 !important; }}
h3 {{ font-size: 1.02rem !important; font-weight: 700 !important; color: {INK}; margin: 0 0 2px 0 !important; padding: 0 !important; }}
/* sidebar */
[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] > div {{ overflow-x: hidden; }}
[data-testid="stSidebarContent"] {{ display: flex; flex-direction: column; }}
[data-testid="stSidebarHeader"] {{ order: 0; height: 2rem; }}
[data-testid="stSidebarUserContent"] {{ order: 1; padding: 0 1.1rem 0.4rem 1.1rem; }}
[data-testid="stSidebarNav"] {{ order: 2; padding: 0 0.5rem; }}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; padding: 0.42rem 0.7rem; }}
[data-testid="stSidebarNav"] a span {{ font-weight: 600; font-size: 14px; color: {INK_2}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {INDIGO_XL}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: {INDIGO}; }}
/* cards */
[class*="st-key-card"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 18px;
    padding: 16px 18px 12px 18px; box-shadow: 0 1px 2px rgba(22,22,42,.04), 0 8px 24px rgba(22,22,42,.05); }}
[class*="st-key-card-kpi"] {{ min-height: 128px; }}
[class*="st-key-filters"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 10px 16px 4px 16px; }}
.eyebrow {{ font-size: 11.5px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: {INDIGO}; }}
.sub {{ color: {INK_2}; font-size: 14px; line-height: 1.5; }}
.muted {{ color: {INK_3}; font-size: 12.5px; }}
.kpi-lab {{ color: {INK_2}; font-size: 13px; font-weight: 600; }}
.kpi-num {{ font-size: 30px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; margin: 6px 0 2px 0; }}
.kpi-ctx {{ color: {INK_3}; font-size: 12.5px; }}
.badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px;
    font-weight: 700; font-size: 12.5px; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.chip {{ display: inline-block; padding: 3px 10px; border-radius: 8px; background: {CANVAS}; color: {INK_2};
    font-size: 12.5px; font-weight: 600; margin: 0 6px 6px 0; border: 1px solid {LINE}; }}
.brand {{ display: flex; align-items: center; gap: 10px; margin: 2px 0 14px 0; }}
.brand-name {{ font-weight: 800; font-size: 15px; color: {INK}; line-height: 1.15; }}
.brand-name span {{ color: {INDIGO}; }}
.side-foot {{ color: {INK_3}; font-size: 12px; margin-top: 10px; padding-top: 10px; border-top: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
div[data-testid="stMetricValue"] {{ font-weight: 800; }}
.hero {{ position: relative; overflow: hidden; border-radius: 24px; padding: 30px 34px 28px 34px; margin-top: 4px;
    background: radial-gradient(1200px 400px at 85% -20%, #6A62E0 0%, rgba(106,98,224,0) 60%),
                linear-gradient(135deg, #14124A 0%, #221E7A 50%, #2F28A8 100%);
    box-shadow: 0 18px 40px rgba(34, 30, 122, 0.28); }}
.hero .pulse {{ position: absolute; left: 0; right: 0; bottom: 6px; width: 100%; height: 120px; opacity: .55; }}
.hero .pulse path {{ stroke-dasharray: 2600; stroke-dashoffset: 2600; animation: beat 5.5s ease-in-out infinite; }}
@keyframes beat {{ 0% {{ stroke-dashoffset: 2600; }} 55% {{ stroke-dashoffset: 0; }} 100% {{ stroke-dashoffset: -2600; }} }}
.hero-grid {{ position: relative; display: grid; grid-template-columns: 1.7fr 1fr; gap: 28px; align-items: center; }}
.hero-eyebrow {{ color: #B9B5FF; font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
.hero-title {{ color: white; font-size: 34px; font-weight: 800; line-height: 1.12; letter-spacing: -0.03em; margin: 10px 0 10px 0; }}
.hero-title span {{ color: #FF9A7A; }}
.hero-sub {{ color: #D6D4FF; font-size: 14.5px; line-height: 1.55; max-width: 640px; }}
.hero-stats {{ display: grid; gap: 10px; }}
.hs {{ background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); border-radius: 16px;
    padding: 10px 16px; backdrop-filter: blur(6px); }}
.hs b {{ color: white; font-size: 24px; font-weight: 800; letter-spacing: -0.02em; }}
.hs b small {{ font-size: 14px; font-weight: 700; color: #D6D4FF; }}
.hs span {{ display: block; color: #C9C7F5; font-size: 12.5px; }}
[class*="st-key-night-card"] {{ border-radius: 22px; padding: 18px 20px 6px 20px;
    background: radial-gradient(900px 380px at 50% 0%, #26236B 0%, #121131 70%);
    box-shadow: 0 14px 36px rgba(18, 17, 49, 0.35); }}
.night-head {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; }}
.night-eyebrow {{ color: #9D98FF; font-size: 11.5px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
.night-title {{ color: white; font-size: 20px; font-weight: 800; letter-spacing: -0.02em; margin-top: 2px; }}
.night-note {{ color: #B5B2E8; font-size: 12.5px; text-align: right; }}
.night-note b {{ color: #FF9A7A; }}
[class*="st-key-card-kpi"] {{ border-top: 3px solid {INDIGO_XL}; }}
.msym {{ font-family: 'Material Symbols Rounded' !important; font-weight: normal; font-style: normal; font-size: 19px;
    line-height: 1; letter-spacing: normal; text-transform: none; white-space: nowrap; font-feature-settings: 'liga'; }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes pop {{ from {{ opacity: 0; transform: scale(.3); transform-box: fill-box; transform-origin: center; }}
                 to {{ opacity: 1; transform: scale(1); transform-box: fill-box; transform-origin: center; }} }}
.kpi2 {{ position: relative; overflow: hidden; border-radius: 18px; padding: 14px 16px 14px 18px; height: 158px;
    background: radial-gradient(240px 150px at 108% -12%, color-mix(in srgb, var(--c) 17%, transparent) 0%, transparent 72%), #fff;
    border: 1px solid {LINE}; box-shadow: 0 1px 2px rgba(22,22,42,.04), 0 8px 24px rgba(22,22,42,.05);
    transition: transform .25s ease, box-shadow .25s ease; animation: rise .6s cubic-bezier(.2,.8,.2,1) both; }}
.kpi2:hover {{ transform: translateY(-4px); box-shadow: 0 18px 34px color-mix(in srgb, var(--c) 22%, transparent); }}
.kpi2::before {{ content: ""; position: absolute; left: 0; top: 16px; bottom: 16px; width: 4px; border-radius: 0 4px 4px 0;
    background: var(--c); }}
.kpi2-top {{ display: flex; align-items: center; gap: 9px; }}
.kpi2-icon {{ width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center; flex: none;
    background: color-mix(in srgb, var(--c) 14%, white); color: var(--c); }}
.kpi2-lab {{ font-size: 13px; font-weight: 700; color: {INK_2}; line-height: 1.25; }}
.kpi2-body {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 10px; margin-top: 10px; }}
.kpi2-num {{ font-size: 30px; font-weight: 800; letter-spacing: -0.025em; color: var(--c); line-height: 1.05; }}
.kpi2-ctx {{ font-size: 12.2px; color: {INK_3}; margin-top: 4px; line-height: 1.35; display: -webkit-box;
    -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }}
.kpi2-vis {{ flex: none; }}
.sec {{ display: flex; gap: 11px; align-items: center; margin-bottom: 8px; }}
.sec-icon {{ width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center; flex: none;
    background: color-mix(in srgb, var(--c) 13%, white); color: var(--c); }}
.sec-title {{ font-size: 16.5px; font-weight: 800; color: {INK}; letter-spacing: -0.015em; }}
[class*="st-key-card"] {{ animation: rise .7s cubic-bezier(.2,.8,.2,1) both; transition: box-shadow .25s ease; }}
[class*="st-key-card"]:hover {{ box-shadow: 0 1px 2px rgba(22,22,42,.04), 0 16px 34px rgba(63,55,201,.10); }}
/* dark panels */
[class*="st-key-dark"] {{ border-radius: 22px; padding: 18px 20px 12px 20px; animation: rise .7s both;
    background: radial-gradient(700px 320px at 80% -10%, #3B35A8 0%, rgba(59,53,168,0) 60%),
                linear-gradient(160deg, #16144A 0%, #1E1B63 60%, #231F78 100%);
    box-shadow: 0 16px 38px rgba(22, 20, 74, 0.32); }}
[class*="st-key-dark"] .sec-title, [class*="st-key-dark"] h3 {{ color: white !important; }}
[class*="st-key-dark"] .muted, [class*="st-key-dark"] .sub {{ color: #BEBBF0 !important; }}
[class*="st-key-dark"] .sub b {{ color: white; }}
[class*="st-key-dark"] .sec-icon {{ background: rgba(255,255,255,.1); color: #C9C6FF; }}
.phead {{ position: relative; overflow: hidden; border-radius: 24px; padding: 24px 28px; display: flex; gap: 24px;
    justify-content: space-between; align-items: center; animation: rise .6s both;
    background: radial-gradient(800px 300px at 90% -30%, var(--glow) 0%, rgba(0,0,0,0) 55%),
                linear-gradient(135deg, #14124A 0%, #221E7A 55%, #2F28A8 100%);
    box-shadow: 0 18px 40px rgba(34, 30, 122, 0.28); }}
.phead-eyebrow {{ color: #B9B5FF; font-size: 12px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }}
.phead-name {{ color: white; font-size: 29px; font-weight: 800; letter-spacing: -0.03em; line-height: 1.12; margin: 6px 0 12px 0; }}
.gchip {{ display: inline-flex; align-items: center; gap: 5px; padding: 4px 11px; border-radius: 999px; margin: 0 6px 6px 0;
    background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.16); color: #E4E2FF; font-size: 12.5px; font-weight: 600; }}
.gchip .msym {{ font-size: 15px; color: #B9B5FF; }}
.phead-r {{ display: flex; align-items: center; gap: 16px; flex: none; }}
.phead-rank {{ color: white; font-size: 30px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; margin-top: 8px; }}
.phead-rank span {{ color: #B9B5FF; font-size: 14px; font-weight: 600; }}
.phead-note {{ color: #C9C7F5; font-size: 12.5px; margin-top: 2px; }}
.cmp {{ position: relative; overflow: hidden; border-radius: 20px; background: #fff; border: 1px solid {LINE};
    padding: 18px 18px 14px 18px; min-height: 360px; box-shadow: 0 8px 24px rgba(22,22,42,.05);
    animation: rise .6s both; transition: transform .25s ease, box-shadow .25s ease; }}
.cmp:hover {{ transform: translateY(-4px); box-shadow: 0 18px 34px color-mix(in srgb, var(--c) 20%, transparent); }}
.cmp::before {{ content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: var(--c); }}
.cmp-name {{ font-weight: 800; font-size: 15.5px; color: {INK}; line-height: 1.25; min-height: 40px; }}
.cmp-row {{ display: flex; justify-content: space-between; padding: 7px 0; border-top: 1px dashed {LINE}; font-size: 13px; color: {INK_2}; }}
.cmp-row b {{ color: {INK}; }}
.step {{ border-radius: 16px; padding: 14px 14px 12px 14px; background: {CANVAS}; border: 1px solid {LINE}; height: 100%; }}
.step-n {{ font-size: 11px; font-weight: 800; color: {INDIGO}; letter-spacing: .1em; }}
.step-t {{ font-size: 14.5px; font-weight: 800; color: {INK}; margin: 4px 0 4px 0; }}
.step-d {{ font-size: 12.8px; color: {INK_2}; line-height: 1.45; }}
.waffles {{ display: flex; justify-content: space-around; gap: 12px; flex-wrap: wrap; padding-top: 6px; }}
.waffle {{ text-align: center; }}
.waffle-n {{ font-size: 24px; font-weight: 800; letter-spacing: -0.02em; margin-top: 10px; }}
.waffle-l {{ font-size: 12.5px; color: {INK_2}; font-weight: 600; }}
.foot {{ color: {INK_3}; font-size: 12px; margin-top: 18px; }}
</style>""", unsafe_allow_html=True)

LOGO = f"""<svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{INDIGO_D}"/>
<stop offset="1" stop-color="#6A62E0"/></linearGradient></defs><rect width="34" height="34" rx="9" fill="url(#g)"/>
<polyline points="6,19 11,19 14,12 18,24 21,16 23,19 28,19" fill="none" stroke="white" stroke-width="2.4"
stroke-linecap="round" stroke-linejoin="round"/></svg>"""


def card(key):
    return st.container(key=f"card-{key}")


def badge(level, text=None):
    return (f'<span class="badge" style="background:{LEVEL_BG[level]};color:{LEVEL_COLOUR[level]}">'
            f'<span class="dot" style="background:{LEVEL_COLOUR[level]}"></span>{text or level + " risk"}</span>')


# ------------------------------------------------------------------ small inline-SVG visuals (animated on load)
_UID = [0]


def _uid():
    _UID[0] += 1
    return f"g{_UID[0]}"


def html(s):
    """Render HTML on one line: Markdown would show indented HTML lines as a code block."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def ring_svg(pct, colour, size=66, stroke=7, text=None, track="#EEEEF6", text_colour=None, sub=None):
    """Progress ring that draws itself (base state = final, so it still shows if animation is off)."""
    r, c = (size - stroke) / 2, size / 2
    circ = 2 * np.pi * r
    off = circ * (1 - max(0.0, min(float(pct), 100.0)) / 100)
    label = ""
    if text is not None:
        dy = -3 if sub else 4.5
        label = (f'<text x="{c}" y="{c + dy:.1f}" text-anchor="middle" font-size="{size * 0.22:.0f}" font-weight="800" '
                 f'fill="{text_colour or colour}" font-family="Plus Jakarta Sans">{text}</text>')
        if sub:
            label += (f'<text x="{c}" y="{c + size * 0.17:.1f}" text-anchor="middle" font-size="{size * 0.1:.0f}" '
                      f'font-weight="600" fill="{text_colour or colour}" opacity=".75" font-family="Plus Jakarta Sans">{sub}</text>')
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<circle cx="{c}" cy="{c}" r="{r:.1f}" fill="none" stroke="{track}" stroke-width="{stroke}"/>'
            f'<circle cx="{c}" cy="{c}" r="{r:.1f}" fill="none" stroke="{colour}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 {c} {c})">'
            f'<animate attributeName="stroke-dashoffset" from="{circ:.1f}" to="{off:.1f}" dur="1.1s" fill="freeze"/></circle>'
            f'{label}</svg>')


def spark_svg(vals, colour, w=118, h=46, zero=True):
    """Area sparkline with a gradient fill, a dashed zero line and a dot on the latest value."""
    v = np.asarray([x for x in vals if pd.notna(x)], float)
    if len(v) < 2:
        return ""
    lo, hi = (min(v.min(), 0), max(v.max(), 0)) if zero else (v.min(), v.max())
    span = (hi - lo) or 1
    xs = np.linspace(3, w - 6, len(v))
    ys = h - 5 - (v - lo) / span * (h - 12)
    gid = _uid()
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"M{xs[0]:.1f},{h} L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + f" L{xs[-1]:.1f},{h} Z"
    zl = ""
    if zero and lo < 0 < hi:
        zy = h - 5 - (0 - lo) / span * (h - 12)
        zl = f'<line x1="0" x2="{w}" y1="{zy:.1f}" y2="{zy:.1f}" stroke="#CFCFE0" stroke-dasharray="2 3"/>'
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><linearGradient id="{gid}" x1="0" x2="0" y1="0" y2="1">'
            f'<stop offset="0" stop-color="{colour}" stop-opacity=".32"/><stop offset="1" stop-color="{colour}" stop-opacity="0"/>'
            f'</linearGradient></defs>{zl}<path d="{area}" fill="url(#{gid})"/>'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.3" stroke-linejoin="round" '
            f'stroke-linecap="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="0">'
            f'<animate attributeName="stroke-dashoffset" from="1" to="0" dur="1.3s" fill="freeze"/></polyline>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3.8" fill="{colour}" stroke="white" stroke-width="1.6"/></svg>')


def mixbar_svg(shares, w=124, h=12):
    """One rounded bar split into the three risk levels (shares in %)."""
    x, parts = 0.0, ""
    for L in LEVELS:
        seg = w * shares.get(L, 0) / 100
        if seg > 0:
            parts += f'<rect x="{x:.1f}" y="0" width="{seg:.1f}" height="{h}" fill="{LEVEL_COLOUR[L]}"/>'
        x += seg
    cid = _uid()
    legend = "".join(f'<tspan fill="{LEVEL_COLOUR[L]}">● </tspan><tspan>{shares.get(L, 0):.0f}%  </tspan>' for L in LEVELS)
    return (f'<svg width="{w}" height="{h + 20}" viewBox="0 0 {w} {h + 20}"><defs><clipPath id="{cid}">'
            f'<rect width="{w}" height="{h}" rx="{h / 2}"/></clipPath></defs><g clip-path="url(#{cid})">{parts}</g>'
            f'<text x="0" y="{h + 15}" font-size="10" font-weight="700" fill="{INK_2}" font-family="Plus Jakarta Sans">{legend}</text></svg>')


def streak_svg(ccn, w=124, years=9):
    """The last few years as dots: coral = lost money, teal = made money."""
    hh = HIST[HIST.ccn == ccn].sort_values("fiscal_year").tail(years)
    if hh.empty:
        return ""
    step = w / years
    dots = "".join(
        f'<circle cx="{step * i + step / 2:.1f}" cy="12" r="{5.2 if i == len(hh) - 1 else 4.4}" '
        f'fill="{CORAL if v < 0 else TEAL}" style="animation:pop .4s {0.06 * i:.2f}s both"/>'
        for i, v in enumerate(hh.total_margin))
    return (f'<svg width="{w}" height="38" viewBox="0 0 {w} 38">{dots}'
            f'<text x="0" y="34" font-size="9.5" fill="{INK_3}" font-family="Plus Jakarta Sans">{int(hh.fiscal_year.iloc[0])}</text>'
            f'<text x="{w}" y="34" font-size="9.5" fill="{INK_3}" text-anchor="end" font-family="Plus Jakarta Sans">'
            f'{int(hh.fiscal_year.iloc[-1])}</text></svg>')


def bullet_svg(value, typical, colour, lo=-0.3, hi=0.25, w=124):
    """Where a value sits on a track, with a tick for the typical US hospital."""
    if pd.isna(value):
        return ""
    pos = lambda v: 6 + (min(max(v, lo), hi) - lo) / (hi - lo) * (w - 12)   # noqa: E731
    return (f'<svg width="{w}" height="40" viewBox="0 0 {w} 40">'
            f'<rect x="6" y="12" width="{w - 12}" height="8" rx="4" fill="#EEEEF6"/>'
            f'<rect x="{pos(lo):.1f}" y="12" width="{pos(0) - pos(lo):.1f}" height="8" rx="4" fill="{CORAL}" opacity=".16"/>'
            f'<line x1="{pos(typical):.1f}" x2="{pos(typical):.1f}" y1="7" y2="25" stroke="{INK_2}" stroke-width="2"/>'
            f'<circle cx="{pos(value):.1f}" cy="16" r="7" fill="{colour}" stroke="white" stroke-width="2.5" '
            f'style="animation:pop .5s .2s both"/>'
            f'<text x="{pos(typical):.1f}" y="37" font-size="9.5" fill="{INK_3}" text-anchor="middle" '
            f'font-family="Plus Jakarta Sans">typical</text></svg>')


def waffle_svg(n, colour, cell=12, gap=4):
    """100 dots, n of them coloured (filled from the bottom), popping in one by one."""
    size = 10 * cell + 9 * gap
    dots = []
    for i in range(100):
        r, c = 9 - i // 10, i % 10
        on = i < n
        dots.append(f'<circle cx="{c * (cell + gap) + cell / 2}" cy="{r * (cell + gap) + cell / 2}" r="{cell / 2}" '
                    f'fill="{colour if on else "#ECECF4"}" style="animation:pop .35s {0.008 * i:.3f}s both"/>')
    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">{"".join(dots)}</svg>'


def kpi(col, key, label, num, ctx, colour=INK, icon="insights", visual="", delay=0):
    """KPI card: tinted icon, label, big number, context line, a small visual on the right, colour glow and accent."""
    with col:
        html(f"""<div class="kpi2" style="--c:{colour};animation-delay:{0.07 * delay:.2f}s">
            <div class="kpi2-top"><span class="kpi2-icon"><span class="msym">{icon}</span></span>
            <span class="kpi2-lab">{label}</span></div>
            <div class="kpi2-body"><div class="kpi2-text"><div class="kpi2-num">{num}</div><div class="kpi2-ctx">{ctx}</div></div>
            <div class="kpi2-vis">{visual}</div></div></div>""")


def section(icon, title, sub=None, colour=INDIGO):
    html(f"""<div class="sec"><span class="sec-icon" style="--c:{colour}"><span class="msym">{icon}</span></span>
        <div><div class="sec-title">{title}</div>{f'<div class="muted">{sub}</div>' if sub else ''}</div></div>""")


def heading(eyebrow, title, sub=None):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if sub:
        st.markdown(f'<div class="sub" style="margin:4px 0 14px 0">{sub}</div>', unsafe_allow_html=True)


def styled(fig, h=320, legend=False):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, hoverlabel=dict(font_family="Plus Jakarta Sans"),
                      font=dict(family="Plus Jakarta Sans", color=INK, size=12.5))
    return fig


def plot(fig, **kw):
    return st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, **kw)


def footer():
    st.markdown('<div class="foot">Built by <b>Isaac Agyapong</b> · Data: CMS Hospital Provider Cost Reports 2011-2023. Scores are statistical estimates '
                'from public reports, not judgments about how any hospital is run. · '
                '<a href="https://github.com/Isaac-Agyapong/Hospital_Financial_Distress_Model">Code and method</a></div>',
                unsafe_allow_html=True)


def open_profile(label):
    st.session_state["hospital"] = label
    st.switch_page(PAGES["profile"])


def current_hospital():
    labels = SCORES.label.tolist()
    if st.session_state.get("hospital") not in labels:
        near = SCORES[SCORES.risk_rank.between(int(0.10 * N), int(0.13 * N)) & SCORES.total_margin.between(-0.05, 0.03)
                      & (SCORES.beds >= 100)]
        st.session_state["hospital"] = near.label.iloc[0] if len(near) else labels[0]
    return st.session_state["hospital"]


def hospital_search(key):
    labels = SCORES.label.tolist()
    cur = current_hospital()
    choice = st.selectbox("Hospital", labels, index=labels.index(cur), key=key, label_visibility="collapsed",
                          placeholder="Search by hospital name, city or state")
    st.session_state["hospital"] = choice
    return SCORES[SCORES.label == choice].iloc[0]


# ================================================================================================= pages
def hero():
    """Deep indigo banner with an animated heartbeat line and the three headline numbers."""
    n_high = int((SCORES.risk_level == "High").sum())
    hm, hr = int(EARLY.loc["XGBoost", "hits"]), int(EARLY.loc["Rule: lowest profit margin first", "hits"])
    html = f"""
<div class="hero">
  <svg class="pulse" viewBox="0 0 1200 160" preserveAspectRatio="none">
    <defs><linearGradient id="pg" x1="0" x2="1"><stop offset="0" stop-color="#7C74FF" stop-opacity="0"/>
      <stop offset=".45" stop-color="#A5A1EA"/><stop offset=".8" stop-color="{CORAL}"/>
      <stop offset="1" stop-color="{CORAL}" stop-opacity="0"/></linearGradient></defs>
    <path d="M0,95 L300,95 L340,95 L360,60 L385,130 L410,30 L440,120 L460,95 L700,95 L730,95 L750,70 L770,112
             L795,48 L820,105 L840,95 L1200,95" fill="none" stroke="url(#pg)" stroke-width="3" stroke-linecap="round"
             stroke-linejoin="round"/>
  </svg>
  <div class="hero-grid">
    <div>
      <div class="hero-eyebrow">Outlook 2024-2025 · machine learning early warning</div>
      <div class="hero-title">{n_high} US hospitals are at <span>high risk</span> of losing money two years in a row</div>
      <div class="hero-sub">Every hospital's latest financial report, scored for the chance of losing money in both of the
        next two years. Explore the map, open any hospital, and test what would change its outlook.</div>
    </div>
    <div class="hero-stats">
      <div class="hs"><b>{N:,}</b><span>hospitals scored</span></div>
      <div class="hs"><b>{HIT}<small> in 100</small></b><span>high-risk picks proved right in the test</span></div>
      <div class="hs"><b>+{hm - hr}</b><span>more early warnings than the best simple rule</span></div>
    </div>
  </div>
</div>"""
    # one line: Markdown would show indented HTML lines as a code block
    st.markdown(" ".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


def night_map(d):
    """Every hospital as a point of light at its real location; high risk glows coral. Click a dot to open it."""
    m = d.merge(LOC, on="ccn", how="left").dropna(subset=["lat"])
    fig = go.Figure()
    layers = [("Lower", 3.2, 0.40, "#8F8BEA"), ("Elevated", 5.0, 0.80, AMBER), ("High", 7.0, 0.95, CORAL)]
    for L, size, alpha, colour in layers:
        s = m[m.risk_level == L]
        if L == "High":     # soft halo behind the high-risk points
            fig.add_scattergeo(lon=s.lon, lat=s.lat, mode="markers", hoverinfo="skip", showlegend=False,
                               marker=dict(size=16, color=colour, opacity=0.13, line=dict(width=0)))
        fig.add_scattergeo(
            lon=s.lon, lat=s.lat, mode="markers", name=f"{L} risk",
            marker=dict(size=size, color=colour, opacity=alpha, line=dict(width=0)),
            customdata=np.stack([s.label, s.risk_level, s.risk_rank, 100 * s.total_margin], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} risk · #%{customdata[2]:,}<br>"
                          "profit margin %{customdata[3]:.1f}%<extra></extra>")
    fig.update_geos(scope="usa", projection_type="albers usa", bgcolor="rgba(0,0,0,0)", showland=True,
                    landcolor="#1C1B4D", showsubunits=True, subunitcolor="#34327C", subunitwidth=0.6,
                    showcountries=False, showlakes=False, showcoastlines=False)
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", x=0.02, y=0.02, font=dict(color="#C9C7F5", size=12),
                                  bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
                      hoverlabel=dict(bgcolor="#16162A", bordercolor="#3F37C9", font=dict(color="white",
                                                                                         family="Plus Jakarta Sans")))
    return fig


def overview():
    hero()
    # clicks remembered from the last run: a dot on the night map opens that hospital, a row in the table too
    ev = st.session_state.get("night")
    pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
    if pts and pts[0].get("customdata"):
        pick = pts[0]["customdata"][0]
        if pick != st.session_state.get("night_applied"):
            st.session_state["night_applied"] = pick
            open_profile(pick)
    st.session_state.setdefault("f_state", "All states")
    st.write("")
    with st.container(key="filters"):
        c = st.columns([1.2, 1.3, 1.6, 1.2])
        states = ["All states"] + sorted(SCORES.state_abbrev.unique())
        state = c[0].selectbox("State", states, key="f_state")
        htype = c[1].pills("Hospital type", ["General", "Critical access"], selection_mode="multi", key="f_type")
        owner = c[2].pills("Owner", ["Nonprofit", "For-profit", "Government"], selection_mode="multi", key="f_owner")
        area = c[3].pills("Area", ["Urban", "Rural"], selection_mode="multi", key="f_area")
    d = SCORES
    if state != "All states":
        d = d[d.state_abbrev == state]
    if htype:
        d = d[d.type_short.isin(htype)]
    if owner:
        d = d[d.ownership.isin(owner)]
    if area:
        d = d[d.rural_urban.isin(area)]
    n = len(d)
    if n == 0:
        st.info("No hospitals match these filters.")
        return
    lv = d.risk_level.value_counts()
    st.write("")
    c = st.columns(4)
    shares = {L: 100 * lv.get(L, 0) / n for L in LEVELS}
    trend = HIST[HIST.ccn.isin(d.ccn)].groupby("fiscal_year").total_margin.median()
    kpi(c[0], "n", "Hospitals in view", f"{n:,}", "latest report, mostly 2023", INDIGO_D, "local_hospital",
        mixbar_svg(shares), 0)
    kpi(c[1], "h", "High risk", f"{lv.get('High', 0):,}",
        f"{TRACK['High']} in 100 like these lost money in the test", CORAL, "warning",
        ring_svg(shares["High"], CORAL, text=f"{shares['High']:.0f}%", track="#FBE3DC"), 1)
    kpi(c[2], "e", "Elevated risk", f"{lv.get('Elevated', 0):,}",
        f"{TRACK['Elevated']} in 100 in the test", AMBER, "trending_up",
        ring_svg(shares["Elevated"], AMBER, text=f"{shares['Elevated']:.0f}%", track="#FBEBD6"), 2)
    kpi(c[3], "m", "Typical profit margin", f"{100 * d.total_margin.median():.1f}%",
        f"typical margin, {int(trend.index.min())}-{int(trend.index.max())}", INDIGO, "payments",
        spark_svg(trend.values, INDIGO), 3)
    st.write("")
    with st.container(key="night-card"):
        st.markdown(f'<div class="night-head"><div><div class="night-eyebrow">The night map</div>'
                    f'<div class="night-title">{n:,} hospitals, each a point of light</div></div>'
                    f'<div class="night-note">Coral = high risk · hover for details · <b>click a dot to open the hospital</b>'
                    f'</div></div>', unsafe_allow_html=True)
        plot(night_map(d), on_select="rerun", selection_mode="points", key="night")
    st.write("")
    left, right = st.columns([1, 1.25])
    with left, card("sun"):
        section("donut_large", "Who is at risk", "Owner, then rural or urban, then risk level. Click a ring to zoom in.")
        g = d.groupby(["ownership", "rural_urban", "risk_level"]).size().reset_index(name="n")
        ids, labels, parents, values, colours = [], [], [], [], []
        own_col = {"Nonprofit": INDIGO, "For-profit": "#6A62E0", "Government": "#2B2596"}
        for o, go_ in g.groupby("ownership"):
            ids.append(o); labels.append(o); parents.append(""); values.append(go_.n.sum()); colours.append(own_col.get(o, INDIGO))
            for a, ga in go_.groupby("rural_urban"):
                ids.append(f"{o}/{a}"); labels.append(a); parents.append(o); values.append(ga.n.sum())
                colours.append(INDIGO_L if a == "Urban" else "#C9C6F4")
                for _, r in ga.iterrows():
                    ids.append(f"{o}/{a}/{r.risk_level}"); labels.append(r.risk_level); parents.append(f"{o}/{a}")
                    values.append(r.n); colours.append(LEVEL_COLOUR[r.risk_level])
        fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
                                    marker=dict(colors=colours, line=dict(color="white", width=1.5)),
                                    insidetextorientation="radial", textfont=dict(size=12),
                                    hovertemplate="<b>%{label}</b><br>%{value:,} hospitals<br>%{percentParent:.0%} of "
                                                  "%{parent}<extra></extra>"))
        plot(styled(fig, 400))
    with right, card("list"):
        top = st.columns([1.7, 1.6])
        with top[0]:
            section("format_list_numbered", "Highest-risk hospitals", "Select a row to open its profile", CORAL)
        lvl = top[1].segmented_control("Show", LEVELS, default="High", key="f_level", label_visibility="collapsed")
        t = d[d.risk_level == lvl] if lvl else d
        table = t.assign(margin=100 * t.total_margin)[["label", "percentile", "margin", "loss_streak"]].head(300)
        ev = st.dataframe(
            table, hide_index=True, use_container_width=True, height=360, on_select="rerun",
            selection_mode="single-row", key="table",
            column_config={
                "label": st.column_config.TextColumn("Hospital", width="large"),
                "percentile": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                "margin": st.column_config.NumberColumn("Margin", format="%.1f%%"),
                "loss_streak": st.column_config.NumberColumn("Years losing")})
        rows = ev.selection.rows if ev is not None else []
        if rows:
            open_profile(table.iloc[rows[0]].label)
    footer()


def swarm(h):
    """All hospitals as dots along the model's risk score, spread vertically by how crowded that score is (a
    beeswarm-like 'comet': most hospitals are low risk, a long tail runs to high risk). This hospital is called out."""
    rng = np.random.default_rng(3)
    x = 100 * SCORES.risk.to_numpy()
    grid = np.linspace(0, 100, 201)
    dens = np.exp(-((x[:, None] - grid[None, :]) / 2.5) ** 2).sum(axis=0)
    width = np.interp(x, grid, dens / dens.max())
    y = rng.uniform(-1, 1, len(x)) * (0.12 + 2.3 * width ** 0.6)
    fig = go.Figure()
    for L in LEVELS[::-1]:
        m = (SCORES.risk_level == L).to_numpy()
        fig.add_scattergl(x=x[m], y=y[m], mode="markers", name=f"{L} risk", hoverinfo="skip",
                          marker=dict(size=4.2, color=LEVEL_COLOUR[L], opacity=0.5 if L == "Lower" else 0.8))
    hx = 100 * h.risk
    fig.add_scatter(x=[hx], y=[0], mode="markers", showlegend=False, hoverinfo="skip",
                    marker=dict(size=34, color=LEVEL_COLOUR[h.risk_level], opacity=0.2))
    fig.add_scatter(x=[hx], y=[0], mode="markers", showlegend=False, hoverinfo="skip",
                    marker=dict(size=15, color=LEVEL_COLOUR[h.risk_level], line=dict(color="white", width=3)))
    fig.add_annotation(x=hx, y=0, ax=0, ay=-78, text=f"<b>{h['name'][:32]}</b><br>score {hx:.0f} · #{int(h.risk_rank):,}",
                       showarrow=True, arrowhead=0, arrowwidth=1.5, arrowcolor=INK, bgcolor="white",
                       bordercolor=LINE, borderpad=6, font=dict(size=12, color=INK, family="Plus Jakarta Sans"))
    for xv, lab in [(100 * Q70, "elevated"), (100 * Q90, "high")]:
        fig.add_vline(x=xv, line_color=INK_3, line_dash="dot", line_width=1)
        fig.add_annotation(x=xv, y=-2.9, text=f"{lab} risk from here", showarrow=False, xanchor="left",
                           font=dict(size=11, color=INK_3))
    fig.update_xaxes(range=[-1, 101], title="model risk score (0-100)", showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, range=[-3.1, 3.4])
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0))
    return styled(fig, 340, legend=True)


def profile():
    heading("Hospital profile", "Check a hospital")
    h = hospital_search("search_profile")
    row = h.to_dict()
    contrib = contributions(row)
    lc = LEVEL_COLOUR[h.risk_level]
    beds = "" if pd.isna(h.beds) else f'<span class="gchip"><span class="msym">bed</span>{int(h.beds):,} beds</span>'
    html(f"""<div class="phead" style="--glow:{lc}55">
        <div><div class="phead-eyebrow">{h.hospital_type} hospital · report year {int(h.fiscal_year)}</div>
        <div class="phead-name">{h["name"]}</div>
        <span class="gchip"><span class="msym">location_on</span>{h.city}, {h.state_abbrev}</span>
        <span class="gchip"><span class="msym">apartment</span>{h.ownership}</span>
        <span class="gchip"><span class="msym">{"park" if h.rural_urban == "Rural" else "location_city"}</span>{h.rural_urban}</span>{beds}</div>
        <div class="phead-r">{ring_svg(h.percentile, lc, size=124, stroke=11, text=f"{h.percentile:.0f}", sub="risk percentile",
                                      track="rgba(255,255,255,.12)", text_colour="white")}
        <div>{badge(h.risk_level)}<div class="phead-rank">#{int(h.risk_rank):,} <span>of {N:,}</span></div>
        <div class="phead-note">riskier than {h.percentile:.0f}% of US hospitals</div></div></div></div>""")
    st.write("")
    c = st.columns(4)
    hh = HIST[HIST.ccn == h.ccn].sort_values("fiscal_year")
    mcol = CORAL if h.total_margin < 0 else TEAL
    kpi(c[0], "pm", "Profit margin", f"{100 * h.total_margin:.1f}%",
        f"US typical {100 * SCORES.total_margin.median():.1f}% · trend since {int(hh.fiscal_year.min())}", mcol, "payments",
        spark_svg(hh.total_margin.values, mcol), 0)
    kpi(c[1], "ls", "Years in a row losing money", f"{int(h.loss_streak)}",
        "made money in its latest year" if h.loss_streak == 0 else f"losing money up to {int(h.fiscal_year)}",
        CORAL if h.loss_streak >= 2 else INDIGO, "event_repeat", streak_svg(h.ccn), 1)
    kpi(c[2], "pc", "Profit on patient care", show_value("operating_margin", h.operating_margin),
        "before gifts and investments", CORAL if (h.operating_margin or 0) < 0 else TEAL, "stethoscope",
        bullet_svg(h.operating_margin, SCORES.operating_margin.median(), CORAL if (h.operating_margin or 0) < 0 else TEAL), 2)
    kpi(c[3], "tr", "Track record of this level", f"{TRACK[h.risk_level]} in 100",
        f"{h.risk_level.lower()}-risk hospitals that then lost money two years (test)", lc, "fact_check",
        ring_svg(TRACK[h.risk_level], lc, text=f"{TRACK[h.risk_level]}", track=LEVEL_BG[h.risk_level]), 3)
    st.write("")
    with card("swarm"):
        section("scatter_plot", "Where it sits among all US hospitals",
                f"Each dot is one of {N:,} hospitals, placed by its risk score. Most are low risk; a long tail runs toward trouble.")
        plot(swarm(h))
    st.write("")
    tabs = st.tabs([":material/insights: Why this score", ":material/bar_chart: Profit history",
                    ":material/groups: Compared with peers"])
    with tabs[0], card("why"):
        section("psychology", "What moved this hospital's score",
                "Each bar is one factor: right = pushes the risk up, left = pulls it down. Exact contributions from the model.",
                CORAL)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
        ylab = []
        for f in top.index:
            typ = SCORES[f].median() if f in META["numeric"] and pd.api.types.is_numeric_dtype(SCORES[f]) else None
            extra = f" · typical {show_value(f, typ)}" if typ is not None and pd.notna(typ) and f not in ("loss", "medicaid_expanded_now") else ""
            who = "that year" if f in ("us_median_margin", "state_median_margin", "state_share_losing") else "this hospital"
            if who == "that year":
                extra = ""
            ylab.append(f"<b>{META['labels'][f].capitalize()}</b><br><span style='color:{INK_3}'>{who} "
                        f"{show_value(f, row[f])}{extra}</span>")
        lim = max(abs(top.values).max() * 1.25, 0.1)
        fig = go.Figure(go.Bar(
            x=top.values, y=ylab, orientation="h", marker=dict(color=[CORAL if v > 0 else TEAL for v in top.values],
                                                            cornerradius=8),
            text=["raises risk" if v > 0 else "lowers risk" for v in top.values], textposition="outside",
            textfont=dict(size=11.5, color=[CORAL if v > 0 else TEAL for v in top.values]),
            hovertemplate="%{y}<extra></extra>"))
        fig.add_vrect(x0=0, x1=lim, fillcolor=CORAL, opacity=0.05, line_width=0, layer="below")
        fig.add_vrect(x0=-lim, x1=0, fillcolor=TEAL, opacity=0.05, line_width=0, layer="below")
        fig.add_vline(x=0, line_color=INK_3, line_width=1.2)
        for x, txt, col in [(lim * 0.97, "RAISES RISK →", CORAL), (-lim * 0.97, "← LOWERS RISK", TEAL)]:
            fig.add_annotation(x=x, y=1.03, yref="paper", yanchor="bottom", text=f"<b>{txt}</b>", showarrow=False,
                               xanchor="right" if x > 0 else "left", font=dict(size=11, color=col))
        fig.update_xaxes(visible=False, range=[-lim, lim])
        fig.update_yaxes(tickfont=dict(size=12.5))
        fig = styled(fig, 480)
        fig.update_layout(bargap=0.32, margin=dict(l=8, r=8, t=34, b=8))
        plot(fig)
    with tabs[1], card("hist"):
        section("bar_chart", "Profit margin by year", "Share of each dollar of income kept as profit. Coral = a loss.")
        fig = go.Figure(go.Bar(x=hh.fiscal_year, y=100 * hh.total_margin,
                               marker=dict(color=[CORAL if v < 0 else INDIGO for v in hh.total_margin], cornerradius=6),
                               text=[f"{100 * v:.1f}%" for v in hh.total_margin], textposition="outside",
                               hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig.add_hrect(y0=min(-5, 100 * hh.total_margin.min() * 1.3), y1=0, fillcolor=CORAL, opacity=0.05, line_width=0)
        fig.update_yaxes(ticksuffix="%", gridcolor=LINE, zerolinecolor=INK_3)
        fig.update_xaxes(dtick=1)
        plot(styled(fig, 380))
    with tabs[2], card("peers"):
        peers = SCORES[(SCORES.hospital_type == h.hospital_type) & (SCORES.state_abbrev == h.state_abbrev)]
        section("groups", f"{h.hospital_type} hospitals in {h.state_abbrev}", "How this hospital compares with its peers")
        pc = st.columns(3)
        ps = 100 * (peers.risk_level == "High").mean()
        kpi(pc[0], "pp1", "Peers in the state", f"{len(peers):,}", f"{h.hospital_type.lower()} hospitals",
            INDIGO, "groups", mixbar_svg({L: 100 * (peers.risk_level == L).mean() for L in LEVELS}), 0)
        kpi(pc[1], "pp2", "Their typical profit margin", f"{100 * peers.total_margin.median():.1f}%",
            f"this hospital {100 * h.total_margin:.1f}%", INDIGO, "payments",
            bullet_svg(h.total_margin, peers.total_margin.median(), mcol), 1)
        kpi(pc[2], "pp3", "Peers at high risk", f"{ps:.0f}%", "of the same type in the state", CORAL, "warning",
            ring_svg(ps, CORAL, text=f"{ps:.0f}%", track="#FBE3DC"), 2)
    footer()


SCENARIOS = {
    "Break even this year": lambda r: {"total_margin": max(r["total_margin"], 0.0),
                                       "operating_margin": max(r["operating_margin"] or 0, 0.0)},
    "Cut agency staff in half": lambda r: {"contract_labor_pct": (r["contract_labor_pct"] or 0) / 2},
    "Add 30 days of cash": lambda r: {"days_cash_on_hand": max(r["days_cash_on_hand"] or 0, 0) + 30},
    "Lose 5 more cents per dollar": lambda r: {"total_margin": r["total_margin"] - 0.05,
                                               "operating_margin": (r["operating_margin"] or 0) - 0.05},
}


def simulator():
    heading("Scenario simulator", "What would change the outlook?",
            "Pick a hospital, try a scenario or move the sliders. The model re-scores the hospital instantly.")
    h = hospital_search("search_sim")
    base = h.to_dict()
    key = f"sim_{h.ccn}"
    defaults = {"m": round(100 * base["total_margin"], 1),
                "o": round(100 * (base["operating_margin"] if pd.notna(base["operating_margin"]) else 0), 1),
                "c": float(round(min(max(base["days_cash_on_hand"] if pd.notna(base["days_cash_on_hand"]) else 0, 0), 365))),
                "a": round(100 * (base["contract_labor_pct"] if pd.notna(base["contract_labor_pct"]) else 0), 1)}
    for k, v in defaults.items():
        st.session_state.setdefault(key + k, v)
    st.write("")
    left, right = st.columns([1.1, 1])
    with left, card("controls"):
        section("auto_awesome", "Try a scenario", "One click applies a realistic change")
        cols = st.columns(2)
        for i, (name, fn) in enumerate(SCENARIOS.items()):
            if cols[i % 2].button(name, key=f"{key}_s{i}", use_container_width=True):
                cur = dict(base, total_margin=st.session_state[key + "m"] / 100,
                           operating_margin=st.session_state[key + "o"] / 100,
                           days_cash_on_hand=st.session_state[key + "c"],
                           contract_labor_pct=st.session_state[key + "a"] / 100)
                ch = fn(cur)
                if "total_margin" in ch:
                    st.session_state[key + "m"] = float(np.clip(round(100 * ch["total_margin"], 1), -40, 40))
                if "operating_margin" in ch:
                    st.session_state[key + "o"] = float(np.clip(round(100 * ch["operating_margin"], 1), -60, 40))
                if "days_cash_on_hand" in ch:
                    st.session_state[key + "c"] = float(min(ch["days_cash_on_hand"], 365))
                if "contract_labor_pct" in ch:
                    st.session_state[key + "a"] = float(round(100 * ch["contract_labor_pct"], 1))
                st.rerun()
        if st.button("Reset to reported numbers", key=f"{key}_reset", type="tertiary", icon=":material/restart_alt:"):
            for k, v in defaults.items():
                st.session_state[key + k] = v
            st.rerun()
        section("tune", "Or adjust the numbers")
        m = st.slider("Profit margin this year", -40.0, 40.0, step=0.5, key=key + "m", format="%.1f%%")
        om = st.slider("Profit on patient care", -60.0, 40.0, step=0.5, key=key + "o", format="%.1f%%")
        cash = st.slider("Days of cash in the bank", 0.0, 365.0, step=1.0, key=key + "c", format="%.0f days")
        agency = st.slider("Agency staff, share of salaries", 0.0, 40.0, step=0.5, key=key + "a", format="%.1f%%")
    row = dict(base, total_margin=m / 100, operating_margin=om / 100, days_cash_on_hand=cash,
               contract_labor_pct=agency / 100, loss=int(m < 0))
    if m >= 0:
        row["loss_streak"] = 0
    elif base["loss"] == 0:
        row["loss_streak"] = 2 if (pd.notna(base["margin_last_year"]) and base["margin_last_year"] < 0) else 1
    if pd.notna(base["margin_last_year"]):
        row["margin_change"] = m / 100 - base["margin_last_year"]
    before, after = predict(base), predict(row)
    pb, pa = percentile_of(before), percentile_of(after)
    lb, la = level_of(before), level_of(after)
    with right, st.container(key="dark-outcome"):
        section("speed", "Outlook", "Grey marker = as reported")
        c = st.columns(2)
        c[0].markdown(f'<div class="muted">As reported</div>{badge(lb)}', unsafe_allow_html=True)
        c[1].markdown(f'<div class="muted">With your changes</div>{badge(la)}', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pa, number=dict(valueformat=".0f", font=dict(size=50, color="white")),
            title=dict(text="risk percentile (100 = highest)", font=dict(size=12.5, color="#BEBBF0")),
            gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 70, 90, 100], tickfont=dict(size=11, color="#BEBBF0"),
                                 tickcolor="#5C58B0"),
                       bar=dict(color=LEVEL_COLOUR[la], thickness=0.3), bgcolor="rgba(255,255,255,.04)",
                       steps=[dict(range=[0, 70], color="rgba(18,165,148,.22)"),
                              dict(range=[70, 90], color="rgba(232,150,46,.25)"),
                              dict(range=[90, 100], color="rgba(228,87,46,.30)")],
                       threshold=dict(line=dict(color="#E9E8FF", width=3), thickness=0.9, value=pb), borderwidth=0)))
        fig = styled(fig, 290)
        fig.update_layout(margin=dict(l=40, r=40, t=46, b=6), font=dict(color="white"))
        plot(fig)
        moved = pa - pb
        arrow = "no change" if abs(moved) < 0.5 else (f"▲ {moved:.0f} points riskier" if moved > 0 else f"▼ {-moved:.0f} points safer")
        colour = "#E9E8FF" if abs(moved) < 0.5 else (CORAL if moved > 0 else "#4FD1C0")
        html(f"""<div style="text-align:center;font-size:20px;font-weight:800;color:{colour}">{arrow}</div>
            <div class="sub" style="text-align:center;margin-top:4px">In the test, <b>{TRACK[la]} in 100</b> hospitals at the
            <b>{la.lower()}</b> level lost money in both of the next two years.</div><div style="height:6px"></div>""")
    st.write("")
    with card("impact"):
        section("bolt", "What each change would do on its own",
                "Starting from this hospital's reported numbers, best change first.", AMBER)
        rows = []
        for name, fn in SCENARIOS.items():
            r = dict(base, **fn(base))
            r["loss"] = int(r["total_margin"] < 0)
            if r["total_margin"] >= 0:
                r["loss_streak"] = 0
            if pd.notna(base["margin_last_year"]):
                r["margin_change"] = r["total_margin"] - base["margin_last_year"]
            rows.append((name, predict(r)))
        rows.sort(key=lambda x: x[1])
        cols = st.columns(len(rows))
        for i, (col, (name, risk)) in enumerate(zip(cols, rows)):
            d = percentile_of(risk) - pb
            lvl = level_of(risk)
            if d <= -10:
                verdict, icon, colour = "Lowers the risk a lot", "trending_down", TEAL
            elif d <= -2:
                verdict, icon, colour = "Lowers the risk a little", "south_east", TEAL
            elif d < 2:
                verdict, icon, colour = "Makes almost no difference", "trending_flat", INK_3
            elif d < 10:
                verdict, icon, colour = "Raises the risk a little", "north_east", CORAL
            else:
                verdict, icon, colour = "Raises the risk a lot", "trending_up", CORAL
            if lvl != lb:
                note = f"Moves it from <b>{lb.lower()}</b> to <b>{lvl.lower()}</b> risk."
            else:
                note = f"It stays at <b>{lb.lower()}</b> risk."
            safer = 100 - percentile_of(risk)
            with col:
                html(f"""<div class="kpi2" style="--c:{colour};height:auto;min-height:200px;animation-delay:{0.07 * i:.2f}s">
                    <div class="kpi2-top"><span class="kpi2-icon"><span class="msym">{icon}</span></span>
                    <span class="kpi2-lab" style="color:{INK}">{name}</span></div>
                    <div style="font-size:19px;font-weight:800;color:{colour};margin:12px 0 10px 0;line-height:1.2">{verdict}</div>
                    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">{badge(lb, lb)}
                    <span class="msym" style="color:{INK_3}">arrow_forward</span>{badge(lvl, lvl)}</div>
                    <div class="kpi2-ctx" style="margin-top:10px">{note} Safer than {safer:.0f}% of US hospitals
                    afterwards.</div></div>""")
    footer()


def compare():
    heading("Compare", "Compare hospitals side by side",
            "Pick up to four hospitals, for example a hospital and its local competitors.")
    labels = SCORES.label.tolist()
    cur = current_hospital()
    same = SCORES[SCORES.state_abbrev == SCORES.loc[SCORES.label == cur, "state_abbrev"].iloc[0]].label.head(3).tolist()
    default = list(dict.fromkeys([cur] + same))[:3]
    picks = st.multiselect("Hospitals", labels, default=default, max_selections=4, label_visibility="collapsed")
    if not picks:
        st.info("Pick at least one hospital.")
        return
    d = SCORES.set_index("label").loc[picks].reset_index()
    cols = st.columns(len(d))
    palette = [INDIGO, CORAL, TEAL, AMBER]
    for i, (col, (_, h)) in enumerate(zip(cols, d.iterrows())):
        lc = LEVEL_COLOUR[h.risk_level]
        hh = HIST[HIST.ccn == h.ccn].sort_values("fiscal_year")
        with col:
            html(f"""<div class="cmp" style="--c:{lc};animation-delay:{0.08 * i:.2f}s">
                <div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start">
                <div><div class="cmp-name"><span style="color:{palette[i]}">●</span> {h["name"]}</div>
                <div class="muted" style="margin:2px 0 8px 0">{h.city}, {h.state_abbrev} · {h.ownership}</div>{badge(h.risk_level)}</div>
                {ring_svg(h.percentile, lc, size=78, stroke=8, text=f"{h.percentile:.0f}", sub="percentile",
                          track=LEVEL_BG[h.risk_level])}</div>
                <div style="font-size:26px;font-weight:800;color:{INK};margin:10px 0 2px 0">#{int(h.risk_rank):,}
                <span class="muted" style="font-size:13px">of {N:,}</span></div>
                <div style="margin:6px 0 4px 0">{spark_svg(hh.total_margin.values, CORAL if h.total_margin < 0 else TEAL, w=250, h=52)}</div>
                <div class="cmp-row"><span>Profit margin</span><b style="color:{CORAL if h.total_margin < 0 else TEAL}">{100 * h.total_margin:.1f}%</b></div>
                <div class="cmp-row"><span>Years losing money</span><b>{int(h.loss_streak)}</b></div>
                <div class="cmp-row"><span>Days of cash</span><b>{show_value("days_cash_on_hand", h.days_cash_on_hand)}</b></div>
                <div class="cmp-row"><span>Biggest factor</span><b>{str(h.reason_1)}</b></div></div>""")
    st.write("")
    with card("cmp_hist"):
        section("show_chart", "Profit margin over time", "Below the line = losing money")
        fig = go.Figure()
        lows = []
        for i, (_, h) in enumerate(d.iterrows()):
            hh = HIST[HIST.ccn == h.ccn].sort_values("fiscal_year")
            lows.append(100 * hh.total_margin.min())
            fig.add_scatter(x=hh.fiscal_year, y=100 * hh.total_margin, mode="lines+markers", name=h["name"][:34],
                            line=dict(color=palette[i], width=3.2, shape="spline"), marker=dict(size=6))
            fig.add_scatter(x=[hh.fiscal_year.iloc[-1]], y=[100 * hh.total_margin.iloc[-1]], mode="markers+text",
                            text=[f"  {100 * hh.total_margin.iloc[-1]:.0f}%"], textposition="middle right",
                            textfont=dict(color=palette[i], size=12.5), showlegend=False, hoverinfo="skip",
                            marker=dict(size=13, color=palette[i], line=dict(color="white", width=2.5)))
        fig.add_hrect(y0=min(min(lows) * 1.15, -5), y1=0, fillcolor=CORAL, opacity=0.06, line_width=0, layer="below")
        fig.add_hline(y=0, line_color=INK_3, line_width=1)
        fig.update_yaxes(ticksuffix="%", gridcolor=LINE, zeroline=False)
        fig.update_xaxes(dtick=1, range=[2010.5, 2024.2])
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        plot(styled(fig, 380, legend=True))
    footer()


def performance():
    heading("Model performance", "How well does it work?",
            "Trained only on older years, then asked about years it had never seen, the way it would be used for real. "
            "Compared with rules a manager could use without a model.")
    b = BT[BT.group == "All hospitals"]
    rule = round(100 * MAIN.loc["Rule: lost money 2+ years in a row", "precision_top"])
    hm, hr = int(EARLY.loc["XGBoost", "hits"]), int(EARLY.loc["Rule: lowest profit margin first", "hits"])
    auc, auc_rule = MAIN.loc["XGBoost", "roc_auc"], MAIN.drop(index=["XGBoost", "Logistic regression"]).roc_auc.max()
    bars = (f'<svg width="96" height="62" viewBox="0 0 96 62">'
            f'<rect x="10" y="{58 - 50}" width="30" height="50" rx="7" fill="{TEAL}" style="animation:pop .5s both"/>'
            f'<rect x="54" y="{58 - 50 * hr / hm:.0f}" width="30" height="{50 * hr / hm:.0f}" rx="7" fill="#CFCBC2" '
            f'style="animation:pop .5s .15s both"/></svg>')
    c = st.columns(4)
    kpi(c[0], "p1", "High-risk picks that were right", f"{HIT} in 100", "2021 reports, outcome 2022-2023", INDIGO,
        "target", ring_svg(HIT, INDIGO, text=f"{HIT}", track=INDIGO_XL), 0)
    kpi(c[1], "p2", "Best rule of thumb", f"{rule} in 100", "already lost money 2 years running", "#8A8478",
        "rule", ring_svg(rule, "#8A8478", text=f"{rule}", track="#EFEDE8"), 1)
    kpi(c[2], "p3", "Early warnings caught", f"+{hm - hr}", f"{hm} vs {hr} among hospitals still making money", TEAL,
        "notifications_active", bars, 2)
    kpi(c[3], "p4", "Ranking accuracy (ROC-AUC)", f"{auc:.2f}", f"best rule {auc_rule:.2f} · 0.5 = guessing", INDIGO_D,
        "leaderboard", ring_svg(100 * (auc - 0.5) / 0.5, INDIGO_D, text=f"{auc:.2f}", track=INDIGO_XL), 3)
    st.write("")
    left, right = st.columns([1.2, 1])
    with left, card("perf_years"):
        section("bar_chart", "Machine learning matched or beat the best rule every year, and far beat picking at random",
                "Of the hospitals each method flagged, how many in 100 went on to lose money in both of the next two years")
        yrs = sorted(b.test_year.unique())
        xs = [f"{y} reports" for y in yrs]
        ml = [100 * b[(b.model == "XGBoost") & (b.test_year == y)].precision_top.iloc[0] for y in yrs]
        rules = b[b.model.str.startswith("Rule")]
        best = [100 * rules[rules.test_year == y].precision_top.max() for y in yrs]
        rnd = [100 * b[(b.model == "XGBoost") & (b.test_year == y)].base_rate.iloc[0] for y in yrs]
        fig = go.Figure()
        for name, vals, colr, tc in [("Machine learning", ml, INDIGO, INDIGO), ("Best rule of thumb", best, "#C9C1B4", "#7D766B"),
                                     ("Picking at random", rnd, "#E6E4F0", INK_3)]:
            fig.add_bar(x=xs, y=vals, name=name, marker=dict(color=colr, cornerradius=7),
                        text=[f"<b>{v:.0f}</b>" for v in vals], textposition="outside", textfont=dict(size=13, color=tc),
                        hovertemplate=name + ": %{y:.0f} in 100<extra></extra>")
        fig.update_yaxes(visible=False, range=[0, 86])
        fig.update_xaxes(tickfont=dict(size=13))
        fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08,
                          legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center", font=dict(size=12.5)))
        plot(styled(fig, 400, legend=True))
    with right, card("perf_levels"):
        section("grid_view", "Out of every 100 hospitals in each group",
                "how many lost money in both of the next two years (2021 test)", CORAL)
        cells = "".join(
            f'<div class="waffle">{waffle_svg(TRACK[L], LEVEL_COLOUR[L], cell=9, gap=3)}'
            f'<div class="waffle-n" style="color:{LEVEL_COLOUR[L]}">{TRACK[L]} in 100</div>'
            f'<div class="waffle-l">{L} risk</div></div>' for L in LEVELS)
        html(f'<div class="waffles">{cells}</div><div style="height:14px"></div>')
    st.write("")
    with card("method"):
        section("science", "How it works", f"Built from public data, tested the way it would be used")
        steps = [("1 · DATA", "Hospital financial reports",
                  "Every Medicare hospital's yearly cost report, 2011-2023, cleaned and checked in PostgreSQL. "
                  "46,946 hospital-years with a known outcome."),
                 ("2 · QUESTION", "Two years of losses?",
                  "From one year's report: will the hospital lose money in both of the next two years?"),
                 ("3 · MODEL", f"XGBoost on {len(FEATURES)} signals",
                  "Profit history, patient-care profit, loss streak, cash, debt, costs, agency staff, staffing, patient mix, "
                  "size, and the state and national picture."),
                 ("4 · TEST", "Years it never saw",
                  "Settings chosen on 2017-2018 only; tested on 2019, 2020 and 2021 reports against rules of thumb.")]
        sc = st.columns(4)
        for col, (n, t_, d_) in zip(sc, steps):
            col.markdown(f'<div class="step"><div class="step-n">{n}</div><div class="step-t">{t_}</div>'
                         f'<div class="step-d">{d_}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="muted" style="margin-top:12px">Limits: hospitals that close stop filing, so closures are not '
                    f'counted; scores drift with the economy, so the app shows risk levels with their track record instead of '
                    f'exact chances. Explanations are exact per-factor contributions from the trees (the same idea as SHAP).</div>',
                    unsafe_allow_html=True)
    footer()


PAGES = {
    "overview": st.Page(overview, title="Overview", icon=":material/space_dashboard:", default=True),
    "profile": st.Page(profile, title="Hospital profile", icon=":material/local_hospital:"),
    "simulator": st.Page(simulator, title="Scenario simulator", icon=":material/tune:"),
    "compare": st.Page(compare, title="Compare", icon=":material/compare_arrows:"),
    "performance": st.Page(performance, title="Performance", icon=":material/verified:"),
}
with st.sidebar:
    st.markdown(f'<div class="brand">{LOGO}<div class="brand-name">Hospital Distress<br><span>Forecast</span></div></div>',
                unsafe_allow_html=True)
st.navigation(list(PAGES.values()), position="sidebar").run()

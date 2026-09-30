"""Hospital Financial Distress Forecast: Streamlit app.

    streamlit run app/app.py

Reads app/data/ (written by Python/02_train_model.py): the latest risk score for every hospital, its top reasons,
the trained XGBoost model (for the "What if?" page) and each hospital's profit history.
Design: soft "bento" layout: lavender-grey canvas, rounded white cards, indigo for the model, coral for risk,
Plus Jakarta Sans, navigation in the sidebar.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

DATA = Path(__file__).parent / "data"
INDIGO, INDIGO_L, CORAL, CORAL_L, AMBER, TEAL = "#3F37C9", "#A5A1EA", "#E4572E", "#F8C9BA", "#F2A541", "#12A594"
INK, INK_2, CANVAS, CARD, LINE = "#1B1B2F", "#62667A", "#F3F3F9", "#FFFFFF", "#E6E6F0"
LEVEL_COLOUR = {"High": CORAL, "Elevated": AMBER, "Lower": TEAL}
MODELS = Path(__file__).resolve().parents[1] / "models"

st.set_page_config(page_title="Hospital Financial Distress Forecast", page_icon="🏥", layout="wide")


# ---------------------------------------------------------------------------------------------------- data
@st.cache_data
def load():
    s = pd.read_csv(DATA / "hospital_scores.csv", dtype={"ccn": str})
    s["label"] = s.hospital_name + " (" + s.city.fillna("").str.title() + ", " + s.state_abbrev + ")"
    dup = s.label.duplicated(keep=False)
    s.loc[dup, "label"] += " #" + s.loc[dup, "ccn"]
    hist = pd.read_csv(DATA / "margin_history.csv", dtype={"ccn": str})
    meta = json.loads((DATA / "features.json").read_text(encoding="utf-8"))
    limits = json.loads((DATA / "winsor_limits.json").read_text(encoding="utf-8"))
    return s.sort_values("risk_rank"), hist, meta, limits


@st.cache_resource
def model():
    m = xgb.XGBClassifier()
    m.load_model(DATA / "xgb_final.json")
    return m


@st.cache_data
def test_results():
    """Numbers quoted on the pages, read from the saved test results so they always match the model."""
    bt = pd.read_csv(MODELS / "backtest.csv")
    t = pd.read_csv(MODELS / "test_predictions_2021.csv")
    q70, q90 = t.risk.quantile(0.7), t.risk.quantile(0.9)
    lvl = np.select([t.risk >= q90, t.risk >= q70], ["High", "Elevated"], "Lower")
    track = {k: round(100 * t.actual[lvl == k].mean()) for k in ("High", "Elevated", "Lower")}
    main = bt[(bt.test_year == 2021) & (bt.group == "All hospitals")].set_index("model")
    early = bt[(bt.test_year == 2021) & (bt.group == "Made money this year")].set_index("model")
    auc = bt[bt.group == "All hospitals"].groupby("model").roc_auc.agg(["min", "max"])
    return track, main, early, auc


SCORES, HIST, META, LIMITS = load()
N = len(SCORES)
Q70, Q90 = SCORES.risk.quantile(0.70), SCORES.risk.quantile(0.90)
TRACK, MAIN, EARLY, AUC = test_results()
P = lambda model: round(100 * MAIN.loc[model, "precision_top"])      # noqa: E731


def level_of(risk):
    return "High" if risk >= Q90 else "Elevated" if risk >= Q70 else "Lower"


def predict(row):
    """Score one hospital's (possibly edited) values exactly as in training."""
    X = pd.DataFrame([row])[META["numeric"] + META["categorical"]].copy()
    for c, (lo, hi) in LIMITS.items():
        X[c] = X[c].astype(float).clip(lo, hi)
    for c in META["numeric"]:
        X[c] = X[c].astype(float)
    for c in META["categorical"]:
        X[c] = pd.Categorical(X[c].astype(str), categories=META["categories"][c])
    return float(model().predict_proba(X)[:, 1][0])


# ---------------------------------------------------------------------------------------------------- style
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
.stApp, .stApp * {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
.stApp {{ background: {CANVAS}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebarNav"] a span {{ font-weight: 600; }}
h1, h2, h3, h1 *, h2 *, h3 * {{ color: {INK}; letter-spacing: -0.02em; font-family: 'Plus Jakarta Sans', sans-serif !important; }}
/* sidebar: brand block above the page links */
[data-testid="stSidebarContent"] {{ display: flex; flex-direction: column; }}
[data-testid="stSidebarUserContent"] {{ order: 1; padding-top: 0.5rem; padding-bottom: 0; }}
[data-testid="stSidebarNav"] {{ order: 2; margin-top: 14px; padding-top: 14px; border-top: 1px solid {LINE}; }}
[data-testid="stSidebarHeader"] {{ order: 0; }}
[class*="st-key-card"] {{ background: {CARD}; border: 1px solid {LINE} !important; border-radius: 22px;
    padding: 18px 20px 14px 20px; box-shadow: 0 6px 24px rgba(40, 40, 90, 0.06); }}
[class*="st-key-hero"] {{ background: linear-gradient(135deg, #2B2596 0%, {INDIGO} 55%, #6A62E0 100%);
    border-radius: 26px; padding: 28px 30px; color: white; box-shadow: 0 12px 30px rgba(63, 55, 201, 0.25); }}
[class*="st-key-hero"] * {{ color: white; }}
.eyebrow {{ font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: {INDIGO}; }}
.big {{ font-size: 44px; font-weight: 800; line-height: 1.05; letter-spacing: -0.03em; }}
.sub {{ color: {INK_2}; font-size: 14.5px; }}
.pill {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 13px; }}
.reason {{ padding: 10px 14px; border-radius: 14px; background: {CANVAS}; margin-bottom: 8px; font-size: 14.5px; }}
.kpi-num {{ font-size: 34px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }}
.kpi-lab {{ color: {INK_2}; font-size: 13.5px; }}
.foot {{ color: {INK_2}; font-size: 12px; margin-top: 24px; }}
</style>""", unsafe_allow_html=True)


def card(key):
    return st.container(border=False, key=f"card-{key}")


def kpi(col, key, num, label, colour=INK):
    with col, card(key):
        st.markdown(f'<div class="kpi-num" style="color:{colour}">{num}</div><div class="kpi-lab">{label}</div>',
                    unsafe_allow_html=True)


def pill(level):
    return (f'<span class="pill" style="background:{LEVEL_COLOUR[level]}22;color:{LEVEL_COLOUR[level]}">'
            f'{level} risk</span>')


def fig_layout(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans", color=INK, size=13))
    return fig


def footer():
    st.markdown('<div class="foot">Built by <b>Isaac Agyapong</b> · Data: CMS Hospital Provider Cost Reports 2011-2023 · '
                'A statistical estimate from public reports, not a judgment about any hospital. · '
                '<a href="https://github.com/Isaac-Agyapong/Hospital_Financial_Distress_Model">Code on GitHub</a></div>',
                unsafe_allow_html=True)


def hospital_picker(key):
    labels = SCORES.label.tolist()
    # first visit: a mid-sized hospital just below the high-risk line with a margin near zero, so the What if?
    # sliders can move it between risk levels
    near = SCORES[SCORES.risk_rank.between(int(0.10 * N), int(0.13 * N)) & SCORES.total_margin.between(-0.05, 0.03)
                  & (SCORES.beds >= 100)]
    start = near.label.iloc[0] if len(near) else labels[int(0.10 * N)]
    default = labels.index(st.session_state.get("hospital", start)) if st.session_state.get("hospital") in labels         else labels.index(start)
    choice = st.selectbox("Search for a hospital by name, city or state", labels, index=default, key=key)
    st.session_state["hospital"] = choice
    return SCORES[SCORES.label == choice].iloc[0]


# ---------------------------------------------------------------------------------------------------- pages
def home():
    n_high = int((SCORES.risk_level == "High").sum())
    with st.container(key="hero"):
        st.markdown(f'<div style="font-size:13px;font-weight:700;letter-spacing:.12em;opacity:.8">HOSPITAL FINANCIAL '
                    f'DISTRESS FORECAST · 2024-2025</div>'
                    f'<div class="big" style="margin:10px 0 8px 0">{n_high} US hospitals are at high risk of losing money '
                    f'two years in a row</div>'
                    f'<div style="font-size:16px;opacity:.9;max-width:820px">A machine learning model read the latest '
                    f'financial report of {N:,} hospitals and ranked how likely each one is to lose money in both of the '
                    f'next two years. When I tested it on years it had never seen, {round(P("XGBoost") / 10)} in 10 of the hospitals '
                    f'it flagged as high risk did.</div>', unsafe_allow_html=True)
    st.write("")
    c = st.columns(4)
    lv = SCORES.risk_level.value_counts()
    kpi(c[0], "k1", f"{lv.get('High', 0):,}", "hospitals at <b>high</b> risk (top 10%)", CORAL)
    kpi(c[1], "k2", f"{lv.get('Elevated', 0):,}", "at <b>elevated</b> risk (next 20%)", AMBER)
    kpi(c[2], "k3", f"{lv.get('Lower', 0):,}", "at <b>lower</b> risk", TEAL)
    kpi(c[3], "k4", f"{P('XGBoost')} in 100", "high-risk picks that then lost money 2 years (tested)", INDIGO)
    st.write("")
    left, right = st.columns([1.55, 1])
    with left, card("map"):
        st.markdown('<div class="eyebrow">Where</div><h3 style="margin:2px 0 0 0">Share of each state\'s hospitals at high risk</h3>',
                    unsafe_allow_html=True)
        st_ = (SCORES.groupby("state_abbrev").agg(n=("ccn", "size"), high=("risk_level", lambda s: (s == "High").mean()))
               .reset_index())
        fig = go.Figure(go.Choropleth(
            locations=st_.state_abbrev, z=100 * st_.high, locationmode="USA-states", marker_line_color="white",
            colorscale=[[0, "#EEEDFB"], [0.35, INDIGO_L], [0.7, "#F08B6B"], [1, CORAL]], zmin=0, zmax=25,
            colorbar=dict(title="% high risk", ticksuffix="%", thickness=12, len=0.7),
            customdata=np.stack([st_.n, (st_.high * st_.n).round()], axis=1),
            hovertemplate="<b>%{location}</b><br>%{z:.0f}% at high risk<br>%{customdata[1]:.0f} of %{customdata[0]} hospitals<extra></extra>"))
        fig.update_geos(scope="usa", bgcolor="rgba(0,0,0,0)", showlakes=False)
        st.plotly_chart(fig_layout(fig, 380), use_container_width=True, config={"displayModeBar": False})
    with right, card("who"):
        st.markdown('<div class="eyebrow">Who</div><h3 style="margin:2px 0 8px 0">High risk by type of hospital</h3>',
                    unsafe_allow_html=True)
        groups = pd.concat([
            SCORES.assign(g=SCORES.ownership.astype(str)),
            SCORES.assign(g=SCORES.rural_urban.astype(str)),
            SCORES.assign(g=SCORES.hospital_type.map({"General acute care": "General hospital",
                                                     "Critical access": "Small rural (critical access)"}))])
        share = groups.groupby("g").risk_level.apply(lambda s: 100 * (s == "High").mean()).sort_values()
        fig = go.Figure(go.Bar(x=share.values, y=share.index, orientation="h", marker_color=INDIGO_L,
                               text=[f"{v:.0f}%" for v in share.values], textposition="outside"))
        fig.update_xaxes(visible=False, range=[0, share.max() * 1.3])
        fig.update_yaxes(tickfont=dict(size=13))
        st.plotly_chart(fig_layout(fig, 360), use_container_width=True, config={"displayModeBar": False})
    st.write("")
    with card("top"):
        st.markdown('<div class="eyebrow">Watch list</div><h3 style="margin:2px 0 8px 0">The 10 hospitals with the highest risk</h3>',
                    unsafe_allow_html=True)
        top = SCORES.head(10).assign(margin=lambda d: (100 * d.total_margin).round(1),
                                     reason=lambda d: d.reason_1.str.capitalize())
        st.dataframe(top[["risk_rank", "label", "margin", "loss_streak", "reason"]].rename(columns={
            "risk_rank": "Rank", "label": "Hospital", "margin": "Profit margin (%)", "loss_streak": "Years losing money",
            "reason": "Biggest reason"}), hide_index=True, use_container_width=True)
    footer()


def check():
    st.markdown('<div class="eyebrow">Check a hospital</div><h1 style="margin-top:0">How likely is it to lose money '
                'two years in a row?</h1>', unsafe_allow_html=True)
    h = hospital_picker("pick_check")
    level = h.risk_level
    left, right = st.columns([1, 1.25])
    with left, card("dial"):
        st.markdown(f'<h3 style="margin:0">{h.hospital_name.title()}</h3><div class="sub">{h.hospital_type} · '
                    f'{h.ownership} · {h.rural_urban} · {str(h.city).title()}, {h.state_abbrev} · report year '
                    f'{int(h.fiscal_year)}</div>', unsafe_allow_html=True)
        pct = 100 * (1 - (h.risk_rank - 1) / N)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pct, number=dict(suffix="", valueformat=".0f", font=dict(size=46)),
            title=dict(text="risk percentile (100 = highest)", font=dict(size=13, color=INK_2)),
            gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 70, 90, 100], tickfont=dict(size=11)),
                       bar=dict(color=INK, thickness=0.18),
                       steps=[dict(range=[0, 70], color="#D5F2EE"), dict(range=[70, 90], color="#FCE6C4"),
                              dict(range=[90, 100], color="#FAD2C5")], borderwidth=0)))
        fig = fig_layout(fig, 250)
        fig.update_layout(margin=dict(l=40, r=40, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div style="text-align:center">{pill(level)}&nbsp; ranked <b>#{int(h.risk_rank):,}</b> of {N:,}</div>'
                    f'<div class="sub" style="text-align:center;margin-top:10px">Of hospitals in the <b>{level.lower()}</b> '
                    f'group in the past, about <b>{TRACK[level]} in 100</b> lost money in both of the next two years.</div>',
                    unsafe_allow_html=True)
    with right, card("why"):
        st.markdown('<div class="eyebrow">Why</div><h3 style="margin:2px 0 10px 0">What the model looked at most</h3>',
                    unsafe_allow_html=True)
        for k in (1, 2, 3):
            up = bool(h[f"reason_{k}_up"])
            arrow, colour, word = ("▲", CORAL, "raises") if up else ("▼", TEAL, "lowers")
            st.markdown(f'<div class="reason"><span style="color:{colour};font-weight:800">{arrow}</span>&nbsp; '
                        f'<b>{str(h[f"reason_{k}"]).capitalize()}</b> <span class="sub">{word} its risk</span></div>',
                        unsafe_allow_html=True)
        c = st.columns(3)
        c[0].metric("Profit margin", f"{100 * h.total_margin:.1f}%")
        c[1].metric("Years in a row losing money", int(h.loss_streak))
        c[2].metric("Days of cash", "n/a" if pd.isna(h.days_cash_on_hand) or h.days_cash_on_hand < 0
                    else f"{h.days_cash_on_hand:.0f}")
    st.write("")
    with card("hist"):
        st.markdown('<div class="eyebrow">Track record</div><h3 style="margin:2px 0 6px 0">Profit margin by year</h3>',
                    unsafe_allow_html=True)
        hh = HIST[HIST.ccn == h.ccn].sort_values("fiscal_year")
        fig = go.Figure(go.Bar(x=hh.fiscal_year, y=100 * hh.total_margin,
                               marker_color=[CORAL if v < 0 else INDIGO for v in hh.total_margin],
                               text=[f"{100 * v:.1f}%" for v in hh.total_margin], textposition="outside"))
        fig.update_yaxes(ticksuffix="%", gridcolor=LINE, zerolinecolor=INK_2)
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig_layout(fig, 300), use_container_width=True, config={"displayModeBar": False})
        st.caption("Indigo = made money, coral = lost money. A margin of 5% means 5 cents of profit on each dollar.")
    footer()


def what_if():
    st.markdown('<div class="eyebrow">What if?</div><h1 style="margin-top:0">Change the numbers and watch the risk move</h1>'
                '<div class="sub">Pick a hospital, then move the sliders. The model re-scores it instantly.</div>',
                unsafe_allow_html=True)
    h = hospital_picker("pick_whatif")
    base = h.to_dict()
    st.write("")
    left, right = st.columns([1.2, 1])
    with left, card("sliders"):
        st.markdown('<h3 style="margin:0 0 6px 0">Adjust this year\'s numbers</h3>', unsafe_allow_html=True)
        key = f"wi_{h.ccn}"
        m = st.slider("Profit margin this year (%)", -40.0, 40.0, float(round(100 * base["total_margin"], 1)), 0.5,
                      key=key + "m") / 100
        om = st.slider("Profit on patient care (%)", -60.0, 40.0,
                       float(round(100 * (base["operating_margin"] if pd.notna(base["operating_margin"]) else 0), 1)),
                       0.5, key=key + "o") / 100
        cash_default = float(base["days_cash_on_hand"]) if pd.notna(base["days_cash_on_hand"]) else 0.0
        cash = st.slider("Days of cash in the bank", 0.0, 365.0, float(min(max(cash_default, 0), 365)), 1.0, key=key + "c")
        agency = st.slider("Agency staff, % of salaries", 0.0, 40.0,
                           float(round(100 * (base["contract_labor_pct"] if pd.notna(base["contract_labor_pct"]) else 0), 1)),
                           0.5, key=key + "a") / 100
    row = dict(base)
    row.update(total_margin=m, operating_margin=om, days_cash_on_hand=cash, contract_labor_pct=agency)
    row["loss"] = int(m < 0)
    if m >= 0:
        row["loss_streak"] = 0
    elif base["loss"] == 0:        # a new loss: one year, or two if last year was a loss too
        row["loss_streak"] = 2 if (pd.notna(base["margin_last_year"]) and base["margin_last_year"] < 0) else 1
    if pd.notna(base["margin_last_year"]):
        row["margin_change"] = m - base["margin_last_year"]
    before, after = predict(base), predict(row)
    lv_before, lv_after = level_of(before), level_of(after)
    with right, card("result"):
        st.markdown('<h3 style="margin:0 0 10px 0">Result</h3>', unsafe_allow_html=True)
        c = st.columns(2)
        c[0].markdown(f'<div class="kpi-lab">As reported</div>{pill(lv_before)}', unsafe_allow_html=True)
        c[1].markdown(f'<div class="kpi-lab">With your changes</div>{pill(lv_after)}', unsafe_allow_html=True)
        rank_after = int((SCORES.risk > after).sum()) + 1
        st.markdown(f'<div class="big" style="margin-top:14px;color:{LEVEL_COLOUR[lv_after]}">#{rank_after:,}'
                    f'<span class="sub" style="font-size:16px"> of {N:,}</span></div>'
                    f'<div class="sub">new place in the risk ranking (was #{int(h.risk_rank):,})</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=["As reported", "With your changes"], y=[100 * before, 100 * after],
                               marker_color=[INDIGO_L, LEVEL_COLOUR[lv_after]],
                               text=[f"{100 * before:.0f}", f"{100 * after:.0f}"], textposition="outside"))
        fig.update_yaxes(visible=False, range=[0, max(100 * max(before, after) * 1.35, 10)])
        st.plotly_chart(fig_layout(fig, 220), use_container_width=True, config={"displayModeBar": False})
        st.caption("Bars show the model's raw score (0-100). Use it to compare, not as an exact chance: the risk "
                   "level is the reliable part.")
    footer()


def watch_list():
    st.markdown('<div class="eyebrow">Watch list</div><h1 style="margin-top:0">Hospitals most likely to struggle in '
                '2024-2025</h1>', unsafe_allow_html=True)
    with card("filters"):
        c = st.columns([1, 1, 1, 1])
        lv = c[0].multiselect("Risk level", ["High", "Elevated", "Lower"], default=["High"])
        states = c[1].multiselect("State", sorted(SCORES.state_abbrev.unique()))
        owner = c[2].multiselect("Owner", sorted(SCORES.ownership.astype(str).unique()))
        area = c[3].multiselect("Rural or urban", sorted(SCORES.rural_urban.astype(str).unique()))
    d = SCORES[SCORES.risk_level.isin(lv or ["High", "Elevated", "Lower"])]
    if states:
        d = d[d.state_abbrev.isin(states)]
    if owner:
        d = d[d.ownership.astype(str).isin(owner)]
    if area:
        d = d[d.rural_urban.astype(str).isin(area)]
    out = d.assign(margin=(100 * d.total_margin).round(1), reason=d.reason_1.str.capitalize())[
        ["risk_rank", "risk_level", "hospital_name", "city", "state_abbrev", "hospital_type", "ownership", "rural_urban",
         "margin", "loss_streak", "reason"]].rename(columns={
        "risk_rank": "Rank", "risk_level": "Risk", "hospital_name": "Hospital", "city": "City", "state_abbrev": "State",
        "hospital_type": "Type", "ownership": "Owner", "rural_urban": "Area", "margin": "Profit margin (%)",
        "loss_streak": "Years losing money", "reason": "Biggest reason"})
    st.write("")
    with card("table"):
        st.markdown(f"**{len(out):,} hospitals**")
        st.dataframe(out, hide_index=True, use_container_width=True, height=520)
        st.download_button("Download this list (CSV)", out.to_csv(index=False).encode(), "hospital_watch_list.csv",
                           "text/csv")
    footer()


def evidence():
    st.markdown('<div class="eyebrow">How good is it?</div><h1 style="margin-top:0">Tested on years it never saw</h1>'
                '<div class="sub">The model was trained on reports up to 2019 and then asked about 2021 reports: which '
                'hospitals would lose money in both 2022 and 2023? I compared it with rules a manager could use without '
                'a model.</div>', unsafe_allow_html=True)
    rows = [("Machine learning (XGBoost)", P("XGBoost"), INDIGO),
            ("Rule: already lost money 2 years running", P("Rule: lost money 2+ years in a row"), "#B7AFA3"),
            ("Logistic regression", P("Logistic regression"), INDIGO_L),
            ("Rule: thinnest profit margin first", P("Rule: lowest profit margin first"), "#B7AFA3")]
    st.write("")
    left, right = st.columns(2)
    with left, card("ev1"):
        st.markdown('<h3 style="margin:0">Of the 10% it flagged, how many did lose money both years?</h3>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(y=[r[0] for r in rows][::-1], x=[r[1] for r in rows][::-1], orientation="h",
                               marker_color=[r[2] for r in rows][::-1], text=[f"{r[1]} in 100" for r in rows][::-1],
                               textposition="outside"))
        fig.update_xaxes(visible=False, range=[0, 100])
        st.plotly_chart(fig_layout(fig, 300), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Across all hospitals, {round(100 * MAIN.loc['XGBoost', 'base_rate'])} in 100 lost money in both 2022 and 2023.")
    with right, card("ev2"):
        st.markdown('<h3 style="margin:0">Early warning: hospitals still making money</h3>', unsafe_allow_html=True)
        hm, hr = int(EARLY.loc["XGBoost", "hits"]), int(EARLY.loc["Rule: lowest profit margin first", "hits"])
        fig = go.Figure(go.Bar(x=["Machine learning", "Best simple rule"], y=[hm, hr],
                               marker_color=[INDIGO, "#B7AFA3"], text=[str(hm), str(hr)], textposition="outside"))
        fig.update_yaxes(visible=False, range=[0, hm * 1.25])
        st.plotly_chart(fig_layout(fig, 300), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"{int(EARLY.loc['XGBoost', 'n']):,} hospitals made money in 2021. Each method picked "
                   f"{int(EARLY.loc['XGBoost', 'flagged'])} of them; bars show how many then lost money in both 2022 "
                   f"and 2023. The model caught {hm - hr} more.")
    st.write("")
    with card("ev3"):
        st.markdown('<h3 style="margin:0 0 6px 0">What each risk level meant in the test</h3>', unsafe_allow_html=True)
        c = st.columns(3)
        for col, (lvl, n) in zip(c, TRACK.items()):
            col.markdown(f'{pill(lvl)}<div class="kpi-num" style="color:{LEVEL_COLOUR[lvl]};margin-top:8px">{n} in 100</div>'
                         f'<div class="kpi-lab">lost money in both of the next two years</div>', unsafe_allow_html=True)
    st.write("")
    with card("ev4"):
        st.markdown(f"""
**How it works, in brief**

- **Data:** the yearly financial report every Medicare hospital files (CMS Hospital Provider Cost Reports, 2011-2023),
  about 4,300 hospitals a year, cleaned in PostgreSQL.
- **Question:** from one year's report, will the hospital lose money in *both* of the next two years?
- **Signals (37):** profit this year and the two before, profit on patient care, years in a row losing money,
  income from investments and gifts, cash, debt, costs, agency staff, staffing, patient mix, size, and how the
  state and the country are doing.
- **Model:** XGBoost (gradient-boosted decision trees). Settings chosen on 2017-2018 only.
- **Tests:** 2019, 2020 and 2021 reports, each predicted by a model trained only on earlier years. The model ranked
  hospitals best in all three (ROC-AUC {AUC.loc["XGBoost", "min"]:.2f}-{AUC.loc["XGBoost", "max"]:.2f}, the best
  rule {AUC.drop(index=["XGBoost", "Logistic regression"])["min"].min():.2f}-{AUC.drop(index=["XGBoost", "Logistic regression"])["max"].max():.2f}).
- **Reasons:** SHAP values show which signals pushed each hospital's score up or down.
- **Limits:** hospitals that close stop filing, so closures are not counted; scores drift with the economy, so the app
  shows risk levels rather than exact chances.
""")
    footer()


pages = [st.Page(home, title="Overview", icon="🏠", default=True),
         st.Page(check, title="Check a hospital", icon="🔎"),
         st.Page(what_if, title="What if?", icon="🎚️"),
         st.Page(watch_list, title="Watch list", icon="📋"),
         st.Page(evidence, title="How good is it?", icon="✅")]
with st.sidebar:
    st.markdown(f'<div style="font-weight:800;font-size:19px;color:{INK};line-height:1.2">Hospital Financial<br>'
                f'<span style="color:{INDIGO}">Distress Forecast</span></div>'
                f'<div class="sub" style="margin:6px 0 4px 0">Machine learning early warning for {N:,} US hospitals</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub" style="margin:10px 0 6px 0">Built by <b>Isaac Agyapong</b> · M.S. Data Science, '
                'Florida Poly</div>', unsafe_allow_html=True)
st.navigation(pages, position="sidebar").run()

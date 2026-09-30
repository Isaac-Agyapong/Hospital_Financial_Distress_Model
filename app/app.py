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


def kpi(col, key, label, num, ctx, colour=INK):
    with col, card(f"kpi-{key}"):
        st.markdown(f'<div class="kpi-lab">{label}</div><div class="kpi-num" style="color:{colour}">{num}</div>'
                    f'<div class="kpi-ctx">{ctx}</div>', unsafe_allow_html=True)


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
def overview():
    heading("Outlook 2024-2025", "Which hospitals are heading for financial trouble?",
            f"Every US hospital's latest financial report, scored by a machine learning model for the chance of "
            f"losing money in <b>both</b> of the next two years. Tested on years it never saw, {HIT} in 100 of its "
            f"high-risk picks did.")
    st.session_state.setdefault("f_state", "All states")
    # a click on the map (stored by the chart from the last run) sets the state filter before it is drawn
    ev = st.session_state.get("map")
    pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
    loc = pts[0].get("location") if pts else None
    if loc and loc != st.session_state.get("map_applied"):
        st.session_state["f_state"] = loc
        st.session_state["map_applied"] = loc
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
    kpi(c[0], "n", "Hospitals in view", f"{n:,}", "latest report, mostly 2023", INK)
    kpi(c[1], "h", "High risk", f"{lv.get('High', 0):,}",
        f"{100 * lv.get('High', 0) / n:.0f}% of view · {TRACK['High']} in 100 of these lost money in the test", CORAL)
    kpi(c[2], "e", "Elevated risk", f"{lv.get('Elevated', 0):,}",
        f"{100 * lv.get('Elevated', 0) / n:.0f}% of view · {TRACK['Elevated']} in 100 in the test", AMBER)
    kpi(c[3], "m", "Typical profit margin", f"{100 * d.total_margin.median():.1f}%",
        f"{100 * d.loss.mean():.0f}% of these hospitals lost money in their latest year", INDIGO)
    st.write("")
    left, right = st.columns([1.45, 1])
    with left, card("map"):
        st.markdown("### Share of hospitals at high risk, by state")
        st.markdown('<div class="muted">Click a state to focus on it</div>', unsafe_allow_html=True)
        g = (d.groupby("state_abbrev").agg(n=("ccn", "size"), high=("risk_level", lambda s: (s == "High").sum()))
             .reset_index().assign(share=lambda x: 100 * x.high / x.n))
        fig = go.Figure(go.Choropleth(
            locations=g.state_abbrev, z=g.share, locationmode="USA-states", marker_line_color="white",
            marker_line_width=1.2, zmin=0, zmax=30,
            colorscale=[[0, "#F1F0FC"], [0.3, INDIGO_L], [0.65, "#EE8D6E"], [1, CORAL]],
            colorbar=dict(ticksuffix="%", thickness=10, len=0.62, x=0.98, outlinewidth=0, tickfont=dict(size=11)),
            customdata=np.stack([g.n, g.high], axis=1),
            hovertemplate="<b>%{location}</b><br>%{z:.0f}% at high risk<br>%{customdata[1]} of %{customdata[0]} "
                          "hospitals<extra></extra>"))
        fig.update_geos(scope="usa", bgcolor="rgba(0,0,0,0)", showlakes=False)
        plot(styled(fig, 330), on_select="rerun", selection_mode="points", key="map")
    with right, card("mix"):
        st.markdown("### Risk mix by owner and area")
        st.markdown('<div class="muted">Share of each group in each risk level</div>', unsafe_allow_html=True)
        groups = [("Nonprofit", d.ownership == "Nonprofit"), ("For-profit", d.ownership == "For-profit"),
                  ("Government", d.ownership == "Government"), ("Urban", d.rural_urban == "Urban"),
                  ("Rural", d.rural_urban == "Rural")]
        rows = [(name, m.sum(), *[100 * (d[m].risk_level == L).mean() if m.sum() else 0 for L in LEVELS])
                for name, m in groups if m.sum() > 0]
        mix = pd.DataFrame(rows, columns=["group", "n"] + LEVELS)[::-1]
        fig = go.Figure()
        for L in LEVELS:
            fig.add_bar(y=mix.group, x=mix[L], name=L, orientation="h", marker_color=LEVEL_COLOUR[L],
                        text=[f"{v:.0f}%" if v >= 9 else "" for v in mix[L]], textposition="inside",
                        insidetextfont=dict(color="white", size=11.5),
                        hovertemplate="%{y}: %{x:.0f}% " + L.lower() + "<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.35,
                          legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=12)))
        fig.update_xaxes(visible=False, range=[0, 100])
        plot(styled(fig, 330, legend=True))
    st.write("")
    with card("list"):
        top = st.columns([3, 1.3])
        top[0].markdown(f"### Highest-risk hospitals in view")
        top[0].markdown('<div class="muted">Select a row to open the hospital\'s profile</div>', unsafe_allow_html=True)
        lvl = top[1].segmented_control("Show", LEVELS, default="High", key="f_level", label_visibility="collapsed")
        t = d[d.risk_level == lvl] if lvl else d
        table = t.assign(margin=100 * t.total_margin, why=t.reason_1.str.capitalize())[
            ["label", "risk_level", "percentile", "margin", "loss_streak", "why"]].head(300)
        ev = st.dataframe(
            table, hide_index=True, use_container_width=True, height=360, on_select="rerun",
            selection_mode="single-row", key="table",
            column_config={
                "label": st.column_config.TextColumn("Hospital", width="large"),
                "risk_level": st.column_config.TextColumn("Risk"),
                "percentile": st.column_config.ProgressColumn("Risk percentile", min_value=0, max_value=100, format="%.0f"),
                "margin": st.column_config.NumberColumn("Profit margin", format="%.1f%%"),
                "loss_streak": st.column_config.NumberColumn("Years losing money"),
                "why": st.column_config.TextColumn("Biggest factor", width="medium")})
        rows = ev.selection.rows if ev is not None else []
        if rows:
            open_profile(table.iloc[rows[0]].label)
    footer()


def profile():
    heading("Hospital profile", "Check a hospital")
    h = hospital_search("search_profile")
    row = h.to_dict()
    contrib = contributions(row)
    with card("head"):
        c = st.columns([2.2, 1])
        with c[0]:
            st.markdown(f'<div style="font-size:22px;font-weight:800;color:{INK};letter-spacing:-0.02em">{h["name"]}</div>'
                        f'<div style="margin:8px 0 2px 0"><span class="chip">{h.hospital_type}</span>'
                        f'<span class="chip">{h.ownership}</span><span class="chip">{h.rural_urban}</span>'
                        f'<span class="chip">{h.city}, {h.state_abbrev}</span>'
                        f'<span class="chip">{0 if pd.isna(h.beds) else int(h.beds):,} beds</span>'
                        f'<span class="chip">report year {int(h.fiscal_year)}</span></div>', unsafe_allow_html=True)
        with c[1]:
            st.markdown(f'<div style="text-align:right">{badge(h.risk_level)}'
                        f'<div style="font-size:30px;font-weight:800;color:{INK};margin-top:6px">#{int(h.risk_rank):,}'
                        f'<span class="muted" style="font-size:14px"> of {N:,}</span></div>'
                        f'<div class="muted">riskier than {h.percentile:.0f}% of US hospitals</div></div>',
                        unsafe_allow_html=True)
    st.write("")
    c = st.columns(4)
    kpi(c[0], "pm", "Profit margin", f"{100 * h.total_margin:.1f}%", f"US typical {100 * SCORES.total_margin.median():.1f}%",
        CORAL if h.total_margin < 0 else TEAL)
    kpi(c[1], "ls", "Years in a row losing money", f"{int(h.loss_streak)}",
        "made money in its latest year" if h.loss_streak == 0 else f"up to {int(h.fiscal_year)}",
        CORAL if h.loss_streak >= 2 else INK)
    kpi(c[2], "pc", "Profit on patient care", show_value("operating_margin", h.operating_margin),
        "before investment income and gifts", INK)
    kpi(c[3], "tr", "Track record of this level", f"{TRACK[h.risk_level]} in 100",
        f"{h.risk_level.lower()}-risk hospitals that then lost money two years (test)", LEVEL_COLOUR[h.risk_level])
    st.write("")
    tabs = st.tabs(["Why this score", "Profit history", "Compared with peers"])
    with tabs[0], card("why"):
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
        labels = [f"{META['labels'][f].capitalize()}: <b>{show_value(f, row[f])}</b>" for f in top.index]
        fig = go.Figure(go.Bar(x=top.values, y=labels, orientation="h",
                               marker_color=[CORAL if v > 0 else TEAL for v in top.values],
                               hovertemplate="%{y}<extra></extra>"))
        fig.add_vline(x=0, line_color=INK_3, line_width=1)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(tickfont=dict(size=13))
        st.markdown("### The factors that moved this hospital's score the most")
        st.markdown(f'<div class="muted"><span style="color:{CORAL};font-weight:700">■ Coral</span> pushes the risk up · '
                    f'<span style="color:{TEAL};font-weight:700">■ teal</span> pulls it down. Longer bar = bigger effect '
                    f'(exact contributions from the model).</div>', unsafe_allow_html=True)
        plot(styled(fig, 380))
    with tabs[1], card("hist"):
        hh = HIST[HIST.ccn == h.ccn].sort_values("fiscal_year")
        st.markdown("### Profit margin by year")
        st.markdown('<div class="muted">Share of each dollar of income kept as profit. Coral = a loss.</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=hh.fiscal_year, y=100 * hh.total_margin,
                               marker_color=[CORAL if v < 0 else INDIGO for v in hh.total_margin],
                               text=[f"{100 * v:.1f}%" for v in hh.total_margin], textposition="outside",
                               hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig.update_yaxes(ticksuffix="%", gridcolor=LINE, zerolinecolor=INK_3)
        fig.update_xaxes(dtick=1)
        plot(styled(fig, 360))
    with tabs[2], card("peers"):
        peers = SCORES[(SCORES.hospital_type == h.hospital_type) & (SCORES.state_abbrev == h.state_abbrev)]
        st.markdown(f"### Where it sits among all {N:,} hospitals")
        st.markdown(f'<div class="muted">Each hospital\'s risk percentile. The marker shows {h["name"]}.</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=SCORES.percentile, nbinsx=50, marker_color=INDIGO_XL,
                                     marker_line=dict(color=INDIGO_L, width=0.5), hoverinfo="skip"))
        fig.add_vrect(x0=70, x1=90, fillcolor=LEVEL_BG["Elevated"], opacity=0.6, line_width=0, layer="below")
        fig.add_vrect(x0=90, x1=100, fillcolor=LEVEL_BG["High"], opacity=0.8, line_width=0, layer="below")
        fig.add_vline(x=h.percentile, line_color=LEVEL_COLOUR[h.risk_level], line_width=3,
                      annotation_text=f"{h['name'][:28]}", annotation_position="top")
        fig.update_xaxes(title="risk percentile (100 = highest risk)", range=[0, 100])
        fig.update_yaxes(visible=False)
        plot(styled(fig, 260))
        c = st.columns(3)
        c[0].metric(f"{h.hospital_type} hospitals in {h.state_abbrev}", f"{len(peers):,}")
        c[1].metric("Their typical profit margin", f"{100 * peers.total_margin.median():.1f}%")
        c[2].metric("Share of them at high risk", f"{100 * (peers.risk_level == 'High').mean():.0f}%")
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
        st.markdown("### Try a scenario")
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
        st.markdown("### Or adjust the numbers")
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
    with right, card("outcome"):
        st.markdown("### Outlook")
        c = st.columns(2)
        c[0].markdown(f'<div class="muted">As reported</div>{badge(lb)}', unsafe_allow_html=True)
        c[1].markdown(f'<div class="muted">With your changes</div>{badge(la)}', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pa, number=dict(valueformat=".0f", font=dict(size=44, color=INK)),
            title=dict(text="risk percentile (100 = highest)", font=dict(size=12.5, color=INK_2)),
            gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 70, 90, 100], tickfont=dict(size=11)),
                       bar=dict(color=LEVEL_COLOUR[la], thickness=0.28),
                       steps=[dict(range=[0, 70], color=LEVEL_BG["Lower"]), dict(range=[70, 90], color=LEVEL_BG["Elevated"]),
                              dict(range=[90, 100], color=LEVEL_BG["High"])],
                       threshold=dict(line=dict(color=INK_3, width=3), thickness=0.9, value=pb), borderwidth=0)))
        fig.update_layout(margin=dict(l=40, r=40, t=40, b=10))
        plot(styled(fig, 280))
        moved = "no change" if abs(pa - pb) < 0.5 else (f"{pa - pb:+.0f} percentile points")
        st.markdown(f'<div class="sub" style="text-align:center">Grey marker = as reported · <b>{moved}</b><br>'
                    f'In the test, {TRACK[la]} in 100 hospitals at the <b>{la.lower()}</b> level lost money in both of '
                    f'the next two years.</div>', unsafe_allow_html=True)
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
    for i, (col, (_, h)) in enumerate(zip(cols, d.iterrows())):
        with col, card(f"cmp{i}"):
            st.markdown(f'<div style="font-weight:800;font-size:15.5px;color:{INK};min-height:44px">{h["name"]}</div>'
                        f'<div class="muted" style="margin-bottom:8px">{h.city}, {h.state_abbrev} · {h.ownership}</div>'
                        f'{badge(h.risk_level)}<div style="font-size:26px;font-weight:800;margin-top:8px">#{int(h.risk_rank):,}'
                        f'<span class="muted"> of {N:,}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sub">Profit margin <b>{100 * h.total_margin:.1f}%</b><br>'
                        f'Years losing money <b>{int(h.loss_streak)}</b><br>'
                        f'Days of cash <b>{show_value("days_cash_on_hand", h.days_cash_on_hand)}</b><br>'
                        f'Biggest factor <b>{str(h.reason_1)}</b></div><div style="height:6px"></div>', unsafe_allow_html=True)
    st.write("")
    with card("cmp_hist"):
        st.markdown("### Profit margin over time")
        fig = go.Figure()
        palette = [INDIGO, CORAL, TEAL, AMBER]
        for i, (_, h) in enumerate(d.iterrows()):
            hh = HIST[HIST.ccn == h.ccn].sort_values("fiscal_year")
            fig.add_scatter(x=hh.fiscal_year, y=100 * hh.total_margin, mode="lines+markers", name=h["name"][:34],
                            line=dict(color=palette[i], width=3, shape="spline"), marker=dict(size=6))
        fig.add_hline(y=0, line_color=INK_3, line_width=1)
        fig.update_yaxes(ticksuffix="%", gridcolor=LINE, zeroline=False)
        fig.update_xaxes(dtick=1)
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        plot(styled(fig, 360, legend=True))
    footer()


def performance():
    heading("Model performance", "How well does it work?",
            "Trained only on older years, then asked about years it had never seen, the way it would be used for real. "
            "Compared with rules a manager could use without a model.")
    b = BT[BT.group == "All hospitals"]
    c = st.columns(4)
    kpi(c[0], "p1", "High-risk picks that were right", f"{HIT} in 100", "2021 reports, outcome 2022-2023", INDIGO)
    kpi(c[1], "p2", "Best rule of thumb", f"{round(100 * MAIN.loc['Rule: lost money 2+ years in a row', 'precision_top'])} in 100",
        "already lost money 2 years running", INK)
    hm, hr = int(EARLY.loc["XGBoost", "hits"]), int(EARLY.loc["Rule: lowest profit margin first", "hits"])
    kpi(c[2], "p3", "Early warnings caught", f"+{hm - hr}", f"{hm} vs {hr} among hospitals still making money", TEAL)
    kpi(c[3], "p4", "Ranking accuracy (ROC-AUC)", f"{MAIN.loc['XGBoost', 'roc_auc']:.2f}",
        f"best rule {MAIN.drop(index=['XGBoost', 'Logistic regression']).roc_auc.max():.2f} · 0.5 = guessing", INK)
    st.write("")
    left, right = st.columns([1.25, 1])
    with left, card("perf_years"):
        st.markdown("### Right picks in the top 10%, every test year")
        st.markdown('<div class="muted">Share of flagged hospitals that went on to lose money in both of the next two years</div>',
                    unsafe_allow_html=True)
        names = {"XGBoost": ("Machine learning", INDIGO, 4), "Rule: lost money 2+ years in a row": ("Rule: already losing 2 years", "#B7AFA3", 2.5),
                 "Logistic regression": ("Logistic regression", INDIGO_L, 2.5), "Rule: lowest profit margin first": ("Rule: thinnest margin", "#D8D2C8", 2.5)}
        fig = go.Figure()
        for mname, (lab, colr, w) in names.items():
            s = b[b.model == mname].sort_values("test_year")
            fig.add_scatter(x=s.test_year.astype(str) + " report", y=100 * s.precision_top, name=lab, mode="lines+markers",
                            line=dict(color=colr, width=w), marker=dict(size=8),
                            hovertemplate=lab + ": %{y:.0f} in 100<extra></extra>")
        base = b[b.model == "XGBoost"].sort_values("test_year")
        fig.add_bar(x=base.test_year.astype(str) + " report", y=100 * base.base_rate, name="All hospitals (base rate)",
                    marker_color="#EEEDF6", hovertemplate="base rate %{y:.0f} in 100<extra></extra>")
        fig.update_yaxes(ticksuffix="%", gridcolor=LINE, range=[0, 80])
        fig.update_layout(legend=dict(orientation="h", y=-0.18))
        plot(styled(fig, 380, legend=True))
    with right, card("perf_levels"):
        st.markdown("### What each risk level meant")
        st.markdown('<div class="muted">2021 test: share that lost money in both of the next two years</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=[f"{L}" for L in LEVELS], y=[TRACK[L] for L in LEVELS],
                               marker_color=[LEVEL_COLOUR[L] for L in LEVELS],
                               text=[f"{TRACK[L]} in 100" for L in LEVELS], textposition="outside"))
        fig.update_yaxes(visible=False, range=[0, 90])
        plot(styled(fig, 380))
    st.write("")
    with card("method"):
        st.markdown(f"""
### Method in brief
- **Data:** the yearly financial report of every Medicare hospital (CMS Hospital Provider Cost Reports, 2011-2023),
  cleaned and checked in PostgreSQL; 46,946 hospital-years with a known outcome.
- **Target:** lost money in *both* of the next two years.
- **Signals ({len(FEATURES)}):** profit this year and the two before, profit on patient care, loss streak, income from
  investments and gifts, cash, debt, costs, agency staff, staffing, patient mix, size, and the state and national picture.
- **Model:** XGBoost, settings chosen on 2017-2018 only; backtests on 2019, 2020 and 2021 reports.
- **Explanations:** exact per-factor contributions from the trees (the same idea as SHAP).
- **Limits:** hospitals that close stop filing, so closures are not counted; scores drift with the economy, so the app
  shows risk levels with their track record instead of exact chances.
""")
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

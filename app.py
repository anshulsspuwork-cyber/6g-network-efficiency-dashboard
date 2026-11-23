"""Network Performance vs Efficiency Analyzer - works with any uploaded CSV/Excel file.

Run:  python -m streamlit run app.py
Needs: streamlit, pandas, numpy, scipy, plotly  (and openpyxl only for .xlsx files)

Upload a file in the sidebar, confirm the column mapping, and every chart, KPI and
filter is built from that file. No data is built in.
"""
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Network vs Efficiency Analyzer", layout="wide")

NONE = "(none)"
QUALITY = ["Low", "Medium", "High"]
LAT, PL, EFF = "Latency", "Packet loss", "Efficiency"
ERR, DEF, SPD = "Error rate", "Defect rate", "Production speed"
MODE, DT = "Operation mode", "Datetime"


@st.cache_data
def read_file(data: bytes, name: str):
    """Load the file and work out which columns suit which kind of dropdown."""
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    else:
        df = pd.read_csv(io.BytesIO(data))
    df.columns = [str(c) for c in df.columns]
    num = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.9]
    few = {c: df[c].nunique(dropna=True) for c in df.columns}
    cat = [c for c in df.columns if 2 <= few[c] <= 8]        # usable as class labels
    small = [c for c in df.columns if 2 <= few[c] <= 20]     # usable as operation mode
    return df, num, cat, small


def tokens(name):
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)  # split camelCase
    return re.findall(r"[a-z]+|\d+", name.lower())


def guess(choices, *keys):
    """Pick the first column whose name contains every word of a key (whole-word match)."""
    for k in keys:
        want = k.split("_")
        for c in choices:
            if all(w in tokens(c) for w in want):
                return c
    return NONE


def rho(a, b):
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan, np.nan
    r, p = stats.spearmanr(a, b)
    return r, p


def fmt(x, spec):
    return "n/a" if pd.isna(x) else format(x, spec)


# ======================= 1. upload =======================
st.sidebar.header("1. Data")
up = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if up is None:
    st.title("Network Performance vs Efficiency Analyzer")
    st.info(
        "Upload a CSV (or Excel) file in the sidebar to begin.\n\n"
        "Minimum needed: a **latency** column, a **packet loss** column and a categorical "
        "**efficiency label** column (e.g. Low / Medium / High). Optional columns unlock more "
        "charts: error rate, defect rate, production speed, operation mode, date and time."
    )
    st.stop()

blob = up.getvalue()
raw, num_cols, cat_cols, small_cols = read_file(blob, up.name)
cols = list(raw.columns)

# ======================= 2. column mapping =======================
with st.sidebar.expander("2. Column mapping", expanded=True):
    def pick(label, choices, *keys):
        """Dropdown limited to columns that fit this field's type; auto-selects a whole-word name match."""
        o = [NONE] + choices
        g = guess(choices, *keys)
        return st.selectbox(label, o, index=o.index(g), key=f"{up.name}-{len(blob)}-{label}")

    c_lat = pick("Latency *", num_cols, "latency")
    c_pl = pick("Packet loss *", num_cols, "packet_loss", "packet")
    eff_mode = st.radio("Efficiency source *", ["Existing label column", "Derive from a numeric column"],
                        key=f"{up.name}-{len(blob)}-effmode")
    if eff_mode == "Existing label column":
        c_eff = pick("Efficiency label * (2-8 classes)", cat_cols, "efficiency", "status")
    else:
        c_eff = pick("Numeric column to rank *", num_cols, "efficiency", "speed", "production")
        higher_better = st.radio("Higher values mean", ["Better efficiency", "Worse efficiency"],
                                 key=f"{up.name}-{len(blob)}-dir") == "Better efficiency"
        n_bins = st.slider("Number of classes", 2, 5, 3)
    c_err = pick("Error rate", num_cols, "error_rate", "error")
    c_def = pick("Defect rate", num_cols, "defect")
    c_spd = pick("Production speed", num_cols, "speed", "production")
    c_mode = pick("Operation mode (2-20 values)", small_cols, "operation_mode", "mode")
    c_date = pick("Date", cols, "date")
    c_time = pick("Time", cols, "timestamp", "time")
    dayfirst = st.checkbox("Dates are day-first (DD-MM-YYYY)", value=True)
    st.caption("* required. Latency is treated as ms and packet loss as %.")

if NONE in (c_lat, c_pl, c_eff):
    st.warning("Map the three required columns (latency, packet loss, efficiency label) in the sidebar.")
    st.stop()

# ======================= build clean table =======================
d = pd.DataFrame(index=raw.index)
d[LAT] = pd.to_numeric(raw[c_lat], errors="coerce")
d[PL] = pd.to_numeric(raw[c_pl], errors="coerce")
if eff_mode == "Existing label column":
    d[EFF] = raw[c_eff].astype("string").str.strip()
else:
    names = {2: ["Low", "High"], 3: ["Low", "Medium", "High"]}.get(n_bins, [f"Q{i + 1}" for i in range(n_bins)])
    if not higher_better:
        names = names[::-1]
    sv = pd.to_numeric(raw[c_eff], errors="coerce")
    okv = sv.notna()
    if okv.sum() < n_bins:
        st.error("Not enough numeric values in that column to make classes.")
        st.stop()
    lab = pd.Series(pd.NA, index=raw.index, dtype="string")
    lab[okv] = pd.qcut(sv[okv].rank(method="first"), n_bins, labels=names).astype(str)
    d[EFF] = lab
for name, col in ((ERR, c_err), (DEF, c_def), (SPD, c_spd)):
    if col != NONE:
        d[name] = pd.to_numeric(raw[col], errors="coerce")
if c_mode != NONE:
    d[MODE] = raw[c_mode].astype(str)
if c_date != NONE:
    txt = raw[c_date].astype(str)
    if c_time not in (NONE, c_date):
        txt = txt + " " + raw[c_time].astype(str)
    dt = pd.to_datetime(txt, dayfirst=dayfirst, errors="coerce")
    if dt.notna().any():
        d[DT] = dt

n_before = len(d)
d = d.dropna(subset=[LAT, PL, EFF])
d[EFF] = d[EFF].astype(str)
if d.empty:
    st.error("No usable rows: latency and packet loss must be numeric and the efficiency label filled in.")
    st.stop()
if n_before - len(d):
    st.sidebar.caption(f"{n_before - len(d):,} rows dropped (missing or non-numeric required values).")

classes = sorted(d[EFF].unique())
if len(classes) > 8:
    st.error(f"'{c_eff}' has {len(classes)} distinct values, so it is a number, not a class label. "
             "In the sidebar, set 'Efficiency source' to 'Derive from a numeric column' to split it into "
             "Low / Medium / High, or pick a column that already holds class labels.")
    st.stop()
rank = {"low": 0, "medium": 1, "high": 2}
default_order = sorted(classes, key=lambda x: (rank.get(x.lower(), 99), x))
ORDER = st.sidebar.multiselect("Efficiency order (worst → best)", classes, default=default_order)
if len(ORDER) < 2:
    st.warning("Select at least two efficiency classes in the order box.")
    st.stop()
d = d[d[EFF].isin(ORDER)]
d["Score"] = d[EFF].map({c: i for i, c in enumerate(ORDER)})

# Network Stability Index: 100 = best, 0 = worst observed (equal weights); bands = terciles
d["NSI"] = 100 * (1 - 0.5 * d[LAT] / max(d[LAT].max(), 1e-9) - 0.5 * d[PL] / max(d[PL].max(), 1e-9))
if len(d) < 3:
    st.error("Need at least 3 rows.")
    st.stop()
d["Network quality"] = pd.qcut(d["NSI"].rank(method="first"), 3, labels=QUALITY)

# ======================= 3. filters =======================
st.sidebar.header("3. Filters")
nq = st.sidebar.multiselect("Network quality", QUALITY, default=QUALITY)
ec = st.sidebar.multiselect("Efficiency class", ORDER, default=ORDER)
f = d[d["Network quality"].isin(nq) & d[EFF].isin(ec)]
if MODE in d:
    om = st.sidebar.selectbox("Operation mode", ["All"] + sorted(d[MODE].unique()))
    if om != "All":
        f = f[f[MODE] == om]
if DT in d:
    d0, d1 = d[DT].min().date(), d[DT].max().date()
    win = st.sidebar.date_input("Time window", (d0, d1), min_value=d0, max_value=d1)
    if isinstance(win, tuple) and len(win) == 2:
        f = f[(f[DT].dt.date >= win[0]) & (f[DT].dt.date <= win[1])]
if f.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.title("Network Performance vs Efficiency")
st.caption(f"File: {up.name} | {len(f):,} of {len(d):,} records after filters")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Network Overview", "Network vs Efficiency", "Quality & Error Impact", "Optimization Insights"])

# ======================= Tab 1 =======================
with tab1:
    c = st.columns(4)
    c[0].metric("Network Stability Index", f"{f['NSI'].mean():.1f} / 100")
    c[1].metric("Mean latency", f"{f[LAT].mean():.2f}")
    c[2].metric("Mean packet loss", f"{f[PL].mean():.2f}")
    c[3].metric("Latency p95", f"{f[LAT].quantile(.95):.2f}")
    a, b = st.columns(2)
    if DT in f:
        daily = f.set_index(DT).resample("D")[[LAT, PL]].mean().dropna().reset_index()
        a.plotly_chart(px.line(daily, x=DT, y=LAT, title="Daily mean latency"), use_container_width=True)
        b.plotly_chart(px.line(daily, x=DT, y=PL, title="Daily mean packet loss"), use_container_width=True)
    a.plotly_chart(px.histogram(f, x=LAT, nbins=50, title="Latency distribution"), use_container_width=True)
    b.plotly_chart(px.histogram(f, x=PL, nbins=50, title="Packet loss distribution"), use_container_width=True)
    st.plotly_chart(px.pie(f, names="Network quality", title="Share of records by network quality"),
                    use_container_width=True)

# ======================= Tab 2 =======================
with tab2:
    lat_rho, lat_p = rho(f[LAT], f["Score"])
    slope = np.nan
    if f[LAT].nunique() > 1 and f["Score"].nunique() > 1:
        slope = stats.linregress(f[LAT], f["Score"]).slope * 10
    c = st.columns(3)
    c[0].metric("Latency Sensitivity Score", fmt(slope, "+.4f"),
                help=f"Change in efficiency score ({ORDER[0]}=0 ... {ORDER[-1]}={len(ORDER) - 1}) per +10 latency units")
    c[1].metric("Latency-efficiency Spearman rho", fmt(lat_rho, "+.4f"))
    c[2].metric("p-value", fmt(lat_p, ".3f"), help="p > 0.05: no statistically detectable relationship")
    ct = (pd.crosstab(f["Network quality"], f[EFF], normalize="index")
          .reindex(columns=ORDER, fill_value=0).reindex(QUALITY).fillna(0).reset_index()
          .melt(id_vars="Network quality", value_name="share", var_name="Efficiency class"))
    st.plotly_chart(px.bar(ct, x="Network quality", y="share", color="Efficiency class", barmode="stack",
                           category_orders={"Efficiency class": ORDER, "Network quality": QUALITY},
                           title="Efficiency mix by network quality"), use_container_width=True)
    bands = pd.cut(f[LAT], 10)
    tbl = pd.crosstab(bands, f[EFF], normalize="index").reindex(columns=ORDER, fill_value=0)
    tbl.index = tbl.index.astype(str)
    st.plotly_chart(px.line(tbl.reset_index().melt(id_vars=LAT, var_name="Efficiency class", value_name="share"),
                            x=LAT, y="share", color="Efficiency class", markers=True,
                            title="Efficiency share by latency band"), use_container_width=True)
    if SPD in f:
        samp = f.sample(min(5000, len(f)), random_state=0)
        st.plotly_chart(px.scatter(samp, x=LAT, y=SPD, color=EFF, opacity=.5, category_orders={EFF: ORDER},
                                   title="Latency vs production speed (up to 5,000-row sample)"),
                        use_container_width=True)

# ======================= Tab 3 =======================
with tab3:
    have = [x for x in (ERR, DEF) if x in f]
    if not have:
        st.info("Map an error-rate or defect-rate column in the sidebar to see this panel.")
    else:
        c = st.columns(3)
        if ERR in f:
            q1, q3 = f[PL].quantile([0.25, 0.75])
            lo, hi = f[f[PL] <= q1][ERR].mean(), f[f[PL] >= q3][ERR].mean()
            ratio = hi / lo if (q1 != q3 and lo) else np.nan
            r1, p1 = rho(f[PL], f[ERR])
            c[0].metric("Packet Loss Impact Ratio", fmt(ratio, ".3f"),
                        help="Mean error rate at worst-quartile packet loss / best quartile. 1.0 = no effect")
            c[1].metric("Packet loss vs error rate (rho)", fmt(r1, "+.4f"), help=f"p = {fmt(p1, '.3f')}")
        if DEF in f:
            r2, p2 = rho(f[PL], f[DEF])
            c[2].metric("Packet loss vs defect rate (rho)", fmt(r2, "+.4f"), help=f"p = {fmt(p2, '.3f')}")
        pb = pd.cut(f[PL], 10)
        g = f.groupby(pb, observed=True)[have].mean()
        g.index = g.index.astype(str)
        st.plotly_chart(px.bar(g.reset_index().melt(id_vars=PL, var_name="Metric", value_name="Mean"),
                               x=PL, y="Mean", color="Metric", barmode="group",
                               title="Mean error / defect rate by packet-loss band"), use_container_width=True)
        if DEF in f:
            st.plotly_chart(px.box(f, x="Network quality", y=DEF, category_orders={"Network quality": QUALITY},
                                   title="Defect rate by network quality"), use_container_width=True)

# ======================= Tab 4 =======================
with tab4:
    pl_rho, pl_p = rho(f[PL], f["Score"])
    sig = [n for n, p in (("latency", lat_p), ("packet loss", pl_p)) if pd.notna(p) and p < 0.05]
    if sig:
        st.warning(f"Statistically detectable relationship with efficiency for: {', '.join(sig)} "
                   f"(latency rho={fmt(lat_rho, '+.3f')}, packet loss rho={fmt(pl_rho, '+.3f')}).")
    else:
        st.info(f"No statistically detectable relationship between the network metrics and efficiency in this "
                f"selection (latency rho={fmt(lat_rho, '+.3f')}, p={fmt(lat_p, '.2f')}; "
                f"packet loss rho={fmt(pl_rho, '+.3f')}, p={fmt(pl_p, '.2f')}).")

    st.subheader("Latency tolerance zones")
    thr = st.slider(f"Flag a band 'at risk' when its share of '{ORDER[0]}' exceeds the overall share by (pp)",
                    0.5, 10.0, 2.0, 0.5)
    worst = f[EFF] == ORDER[0]
    base = worst.mean()
    bands = pd.cut(f[LAT], 10)
    zone = worst.groupby(bands, observed=True).mean().rename("share").reset_index()
    zone[LAT] = zone[LAT].astype(str)
    zone["Status"] = np.where((zone["share"] - base) * 100 > thr, "At risk", "Within tolerance")
    st.plotly_chart(px.bar(zone, x=LAT, y="share", color="Status",
                           title=f"Share of '{ORDER[0]}' efficiency by latency band"), use_container_width=True)
    st.write(f"Overall '{ORDER[0]}' share: **{base:.1%}**. Bands flagged: "
             f"**{(zone['Status'] == 'At risk').sum()}** of {len(zone)}.")

    if ERR in f:
        st.subheader("Packet-loss risk zones")
        pb = pd.cut(f[PL], 10)
        pz = f.groupby(pb, observed=True)[ERR].mean()
        pz.index = pz.index.astype(str)
        st.plotly_chart(px.bar(pz.reset_index(), x=PL, y=ERR, title="Mean error rate by packet-loss band"),
                        use_container_width=True)

    st.subheader("Which columns move with efficiency?")
    num = raw.loc[f.index].select_dtypes("number")
    num = num.loc[:, num.nunique() > 1]
    if f["Score"].nunique() > 1:
        corr = num.corrwith(f["Score"], method="spearman").dropna().reset_index()
    else:
        corr = pd.DataFrame(columns=["Column", "rho"])
    corr.columns = ["Column", "Spearman rho with efficiency"]
    st.plotly_chart(px.bar(corr, x="Column", y="Spearman rho with efficiency",
                           title="Correlation of every numeric column with the efficiency score"),
                    use_container_width=True)
    st.caption("Bars near 0 mean no relationship. If the network columns are near 0 while another column is large, "
               "that column, not the network, is what separates the efficiency classes.")
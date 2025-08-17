# streamlit_app.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import io

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Financial Dashboard", layout="wide")
st.title("📊 AI Revenue & R&D Dashboard")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("ai_financial_market_daily_realistic_synthetic.csv")
    df = df.rename(columns=str.strip)
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = ["R&D_Spending_USD_Mn", "AI_Revenue_USD_Mn",
                "AI_Revenue_Growth_%", "Stock_Impact_%"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Date").set_index("Date")
    df["RD_to_AI_ratio"] = df["R&D_Spending_USD_Mn"] / df["AI_Revenue_USD_Mn"].replace(0, np.nan)
    return df

df = load_data()

# ---------- SIDEBAR ----------
st.sidebar.header("Controls")

# Company selector
all_companies = df["Company"].unique().tolist()
selected_companies = st.sidebar.multiselect(
    "Company(s)", all_companies, default=all_companies
)

# Date range
date_range = st.sidebar.date_input(
    "Date range",
    value=[df.index.min().date(), df.index.max().date()],
    min_value=df.index.min().date(),
    max_value=df.index.max().date()
)
show_forecast = st.sidebar.checkbox("Show 90-day Prophet forecast", value=False)

# Combined filtering
mask = (
    (df.index >= pd.to_datetime(date_range[0]))
    & (df.index <= pd.to_datetime(date_range[1]))
    & (df["Company"].isin(selected_companies))
)
d = df.loc[mask]

# ---------- KPI CARDS ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest AI Revenue", f"${d['AI_Revenue_USD_Mn'].iloc[-1]:,.1f} M")
col2.metric("Latest R&D", f"${d['R&D_Spending_USD_Mn'].iloc[-1]:,.1f} M")
col3.metric("Avg Growth (%)", f"{d['AI_Revenue_Growth_%'].mean():.1f} %")
col4.metric("Events", d["Event"].notna().sum())

# ---------- MAIN CHARTS ----------
st.subheader("Time-Series Overview")
fig = go.Figure()
fig.add_trace(go.Scatter(x=d.index, y=d["R&D_Spending_USD_Mn"],
                         mode='lines', name='R&D Spending ($M)', line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=d.index, y=d["AI_Revenue_USD_Mn"],
                         mode='lines', name='AI Revenue ($M)', line=dict(color='seagreen'), yaxis='y2'))
fig.update_layout(height=400, xaxis_title="Date", yaxis=dict(title="R&D ($M)"),
                  yaxis2=dict(title="AI Revenue ($M)", overlaying='y', side='right'))
st.plotly_chart(fig, use_container_width=True)

# ---------- EVENTS ----------
st.subheader("Events vs Stock Impact")
events = d[d["Event"].notna()]
if not events.empty:
    fig2 = px.scatter(events, x="AI_Revenue_Growth_%", y="Stock_Impact_%",
                      color="Event", size="AI_Revenue_USD_Mn",
                      hover_data=[events.index.date])
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No events in selected range.")

# ---------- MONTHLY HEATMAP ----------
st.subheader("Monthly Averages")
monthly = (d.groupby(d.index.to_period("M"))
             [["R&D_Spending_USD_Mn", "AI_Revenue_USD_Mn",
               "AI_Revenue_Growth_%", "Stock_Impact_%"]]
             .mean()
             .rename(index=str))      # <- convert PeriodIndex → str

fig3 = px.imshow(monthly.T,
                 labels=dict(x="Month", y="Metric", color="Value"),
                 aspect="auto",
                 color_continuous_scale="viridis")
fig3.update_layout(height=300)
st.plotly_chart(fig3, use_container_width=True)

# ---------- PROPHET FORECAST ----------
if show_forecast:
    st.subheader("90-Day Prophet Forecast (Monthly)")
    monthly_prop = d["AI_Revenue_USD_Mn"].resample("M").mean().dropna().reset_index()
    monthly_prop = monthly_prop.rename(columns={"Date": "ds", "AI_Revenue_USD_Mn": "y"})
    if len(monthly_prop) > 20:
        m = Prophet(daily_seasonality=False, weekly_seasonality=False)
        m.fit(monthly_prop)
        future = m.make_future_dataframe(periods=3, freq="M")
        fcst = m.predict(future)
        fig4 = plot_plotly(m, fcst)
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Not enough monthly data to fit Prophet in this range.")


# ---------- GROWTH TREND ----------
st.subheader("AI Revenue Growth Trend")
growth_window = st.sidebar.slider("Growth smoothing (days)", 7, 90, 30)

d["AI_Revenue_Growth_smooth"] = (
    d["AI_Revenue_Growth_%"]
    .rolling(growth_window, min_periods=1)
    .mean()
)

fig_growth = go.Figure()
fig_growth.add_trace(
    go.Scatter(
        x=d.index,
        y=d["AI_Revenue_Growth_%"],
        mode="lines",
        name="Daily",
        line=dict(color="lightgray"),
    )
)
fig_growth.add_trace(
    go.Scatter(
        x=d.index,
        y=d["AI_Revenue_Growth_smooth"],
        mode="lines",
        name=f"{growth_window}-day MA",
        line=dict(color="red", width=2),
    )
)
fig_growth.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Growth %",
    hovermode="x unified",
)
st.plotly_chart(fig_growth, use_container_width=True)


# ---------- DOWNLOAD ----------
st.sidebar.subheader("Export")
csv = d.to_csv().encode()
st.sidebar.download_button("Download cleaned data", data=csv,
                           file_name="openai_cleaned.csv", mime="text/csv")


import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="DRM Dashboard", layout="wide")

st.title("Derivatives Risk Management Dashboard")
st.write("Interactive analysis of LT (Large Cap) and AFFLE (Small Cap)")

data = pd.read_excel("DRM_Project_Output.xlsx")

data["Date"] = pd.to_datetime(data["Date"])


st.subheader("Market Risk Summary")

lt_returns = np.log(data["LT"] / data["LT"].shift(1)).dropna()
affle_returns = np.log(data["AFFLE"] / data["AFFLE"].shift(1)).dropna()

lt_vol = lt_returns.std() * np.sqrt(252)
affle_vol = affle_returns.std() * np.sqrt(252)

col1, col2 = st.columns(2)

col1.metric("LT Annual Volatility", f"{lt_vol:.2%}")
col2.metric("AFFLE Annual Volatility", f"{affle_vol:.2%}")


st.sidebar.header("Dashboard Controls")

stock_choice = st.sidebar.selectbox(
    "Select Stock",
    ["LT", "AFFLE"]
)

min_date = data["Date"].min()
max_date = data["Date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

filtered_data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]

st.metric("Total Observations", len(filtered_data))

st.divider()


st.subheader("Stock Price Trend")

price_col = "LT" if stock_choice == "LT" else "AFFLE"

fig_price = px.line(
    filtered_data,
    x="Date",
    y=price_col,
    title=f"{stock_choice} Price Movement"
)

st.plotly_chart(fig_price, use_container_width=True)

st.divider()


returns = np.log(filtered_data[price_col] / filtered_data[price_col].shift(1)).dropna()


st.subheader("Return Distribution")

fig_hist = px.histogram(
    returns,
    nbins=40,
    title=f"{stock_choice} Log Return Distribution"
)

st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

st.subheader("Rolling Volatility")

rolling_vol = returns.rolling(20).std()

vol_df = pd.DataFrame({
    "Date": filtered_data["Date"].iloc[1:],
    "Volatility": rolling_vol
})

fig_vol = px.line(
    vol_df,
    x="Date",
    y="Volatility",
    title="20 Day Rolling Volatility"
)

st.plotly_chart(fig_vol, use_container_width=True)

st.divider()


st.subheader("Cumulative Returns Comparison")

lt_returns = np.log(filtered_data["LT"] / filtered_data["LT"].shift(1)).fillna(0)
affle_returns = np.log(filtered_data["AFFLE"] / filtered_data["AFFLE"].shift(1)).fillna(0)

cum_lt = lt_returns.cumsum()
cum_affle = affle_returns.cumsum()

cum_df = pd.DataFrame({
    "Date": filtered_data["Date"],
    "LT": cum_lt,
    "AFFLE": cum_affle
})

fig_cum = px.line(
    cum_df,
    x="Date",
    y=["LT", "AFFLE"],
    title="Cumulative Return Comparison"
)

st.plotly_chart(fig_cum, use_container_width=True)

st.divider()

st.subheader("Performance Metrics")

risk_free_rate = 0.06

mean_return = returns.mean()
volatility = returns.std()

sharpe_ratio = (mean_return - risk_free_rate/252) / volatility

col1, col2, col3 = st.columns(3)

col1.metric("Average Daily Return", round(mean_return,5))
col2.metric("Daily Volatility", round(volatility,5))
col3.metric("Sharpe Ratio", round(sharpe_ratio,3))

st.divider()


if "Mispricing" in data.columns:

    st.subheader("Futures Mispricing Analysis")

    fig_mis = px.line(
        filtered_data,
        x="Date",
        y="Mispricing",
        title="Difference Between Market Futures Price and Theoretical Price"
    )

    st.plotly_chart(fig_mis, use_container_width=True)

st.divider()


st.subheader("Dataset Preview")


st.dataframe(filtered_data.tail(10))

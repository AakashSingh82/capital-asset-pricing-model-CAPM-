import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="CAPM - Indian Market", layout="wide")

st.title("CAPM  — Indian Market 🇮🇳")
st.write("Upload tickers (NSE tickers use `.NS`, e.g. `RELIANCE.NS`) or type comma-separated tickers. Default benchmark: NIFTY 50 ( `^NSEI` ).")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Inputs")
    tickers_input = st.text_input("Asset tickers (comma separated)", value="RELIANCE.NS, TCS.NS")
    benchmark = st.text_input("Benchmark ticker", value="^NSEI")
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01").date())
    end_date = st.date_input("End date", value=date.today())
    freq = st.selectbox("Return frequency", options=["Daily", "Monthly", "Yearly"], index=1)
    risk_free = st.number_input("Risk-free rate (annual %, e.g. 7.0)", min_value=0.0, value=7.0, step=0.1)
    show_regression = st.checkbox("Show regression lines on scatter", value=True)
    download_csv = st.checkbox("Show download buttons", value=True)

st.markdown("---")

# --- Helper functions ---
@st.cache_data
def fetch_price_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, group_by='ticker', threads=True, progress=False)
    price = pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                price[t] = raw[(t, 'Adj Close')]
            except Exception:
                price[t] = raw[(t, 'Close')]
    else:
        col = 'Adj Close' if 'Adj Close' in raw.columns else 'Close'
        price[tickers[0]] = raw[col]
    price.columns = [c.strip() for c in price.columns]
    price = price.dropna(how='all')
    return price


def compute_returns(price_df, freq):
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)

    if freq == "Daily":
        ret = price_df.pct_change().dropna()
    elif freq == "Monthly":
        monthly = price_df.resample('M').last()
        ret = monthly.pct_change().dropna()
    elif freq == "Yearly":
        yearly = price_df.resample('Y').last()
        ret = yearly.pct_change().dropna()
    else:
        ret = price_df.pct_change().dropna()

    return ret


def compute_capm(ret_asset, ret_market, rf_annual, periods_per_year):
    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess_asset = ret_asset - rf_period
    excess_market = ret_market - rf_period
    beta = np.polyfit(excess_market, excess_asset, 1)[0]
    alpha = np.polyfit(excess_market, excess_asset, 1)[1]
    exp_market = ret_market.mean()
    expected_return_period = rf_period + beta * (exp_market - rf_period)
    expected_return_annual = (1 + expected_return_period) ** periods_per_year - 1
    alpha_annual = (1 + alpha) ** periods_per_year - 1 if alpha > -1 else alpha * periods_per_year
    return {
        'beta': float(beta),
        'alpha_period': float(alpha),
        'expected_return_annual': float(expected_return_annual),
        'rf_period': float(rf_period)
    }

# --- Main logic ---
if tickers_input.strip() == "":
    st.warning("Please provide at least one asset ticker (comma separated). Use `.NS` for NSE tickers, e.g. `RELIANCE.NS`).")
else:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    all_tickers = list(dict.fromkeys(tickers + [benchmark.strip().upper()]))

    with st.spinner("Fetching price data from Yahoo Finance..."):
        try:
            price = fetch_price_data(all_tickers, start_date, end_date)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    st.subheader("Price chart")
    fig = go.Figure()
    for col in price.columns:
        fig.add_trace(go.Scatter(x=price.index, y=price[col], name=col))
    fig.update_layout(height=450, xaxis_title='Date', yaxis_title='Price (INR)')
    st.plotly_chart(fig, use_container_width=True)

    periods_per_year = 252 if freq == "Daily" else 12 if freq == "Monthly" else 1
    returns = compute_returns(price, freq)

    st.subheader("CAPM calculations")
    market_col = benchmark.strip().upper()
    if market_col not in returns.columns:
        st.error(f"Benchmark {market_col} data not available for the chosen date range.")
        st.stop()

    results = []
    summary = []
    for asset in tickers:
        if asset not in returns.columns:
            st.warning(f"{asset}: No return data available - skipped.")
            continue
        res = compute_capm(returns[asset].values, returns[market_col].values, risk_free / 100.0, periods_per_year)
        results.append((asset, res))
        summary.append({
            'Asset': asset,
            'Beta': round(res['beta'], 4),
            'Expected Return (annual %)': round(res['expected_return_annual'] * 100, 2),
        })

    if not results:
        st.info("No assets had valid data to compute CAPM.")
        st.stop()

    df_summary = pd.DataFrame(summary).set_index('Asset')
    st.dataframe(df_summary)

    # --- Enhanced Section ---
    st.markdown("---")
    st.subheader("Detailed CAPM Metrics")

    # Compute normalized prices for visual comparison
    norm_prices = price / price.iloc[0]
    fig_norm = go.Figure()
    for col in norm_prices.columns:
        fig_norm.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col], name=col))
    fig_norm.update_layout(height=400, xaxis_title='Date', yaxis_title='Normalized Price (Base=1)')
    st.plotly_chart(fig_norm, use_container_width=True)

    # Market stats
    market_return_annual = (1 + returns[market_col].mean()) ** periods_per_year - 1
    market_risk_annual = returns[market_col].std() * np.sqrt(periods_per_year)

    detailed = []
    for asset, res in results:
        vol_annual = returns[asset].std() * np.sqrt(periods_per_year)
        detailed.append({
            'Asset': asset,
            'Beta': round(res['beta'], 3),
            'Expected Return (Annual %)': round(res['expected_return_annual'] * 100, 2),
            'Volatility (Annual %)': round(vol_annual * 100, 2),
            'Alpha (period)': round(res['alpha_period'], 5)
        })

    df_detailed = pd.DataFrame(detailed).set_index('Asset')

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Market Return (Annual %)", f"{market_return_annual * 100:.2f}%")
    with col2:
        st.metric("Market Risk (Volatility %)", f"{market_risk_annual * 100:.2f}%")

    st.dataframe(df_detailed)

    st.markdown("---")
    cols = st.columns(2)
    for i, (asset, res) in enumerate(results):
        with cols[i % 2]:
            st.markdown(f"**{asset} vs {market_col}**")
            df_scatter = pd.DataFrame({
                'asset': returns[asset],
                'market': returns[market_col]
            }).dropna()
            try:
                fig2 = px.scatter(df_scatter, x='market', y='asset', trendline='ols' if show_regression else None,
                                  labels={'market': f'{market_col} returns', 'asset': f'{asset} returns'},
                                  title=f'{asset} (β={res["beta"]:.3f})')
            except Exception:
                fig2 = px.scatter(df_scatter, x='market', y='asset',
                                  labels={'market': f'{market_col} returns', 'asset': f'{asset} returns'},
                                  title=f'{asset} (β={res["beta"]:.3f})')
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Download data")
    if download_csv:
        csv = price.to_csv().encode('utf-8')
        st.download_button("Download price data (CSV)", data=csv, file_name="prices.csv", mime='text/csv')

    st.markdown("Aakash Singh © 2025 · [GitHub]")

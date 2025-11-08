import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="CAPM PRO - Indian Market", layout="wide")

st.title("FinVision — CAPM & Market Analyzer (Indian Market) 🇮🇳")
st.write("Use `.NS` for NSE tickers (e.g. RELIANCE.NS). Default benchmark: NIFTY 50 (`^NSEI`).")

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Inputs")
    tickers_input = st.text_input("Asset tickers (comma separated)", value="RELIANCE.NS, TCS.NS")
    benchmark = st.text_input("Benchmark ticker", value="^NSEI")
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01").date())
    end_date = st.date_input("End date", value=date.today())
    freq = st.selectbox("Return frequency", options=["Daily", "Monthly", "Yearly"], index=1)
    risk_free = st.number_input("Risk-free rate (annual %, e.g. 7.0)", min_value=0.0, value=7.0, step=0.1)
    show_regression = st.checkbox("Show regression lines (requires statsmodels)", value=True)
    download_csv = st.checkbox("Show download buttons", value=True)
    st.markdown("---")
    st.write("💡 Use 3–6 tickers for portfolio optimization demo.")

# --- Helper Functions ---
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
        ret = price_df.resample('M').last().pct_change().dropna()
    elif freq == "Yearly":
        ret = price_df.resample('Y').last().pct_change().dropna()
    return ret


def compute_capm(ret_asset, ret_market, rf_annual, periods_per_year):
    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess_asset = ret_asset - rf_period
    excess_market = ret_market - rf_period
    try:
        beta = np.polyfit(excess_market, excess_asset, 1)[0]
        alpha = np.polyfit(excess_market, excess_asset, 1)[1]
    except Exception:
        beta = np.nan
        alpha = np.nan
    exp_market = np.nanmean(ret_market)
    expected_return_period = rf_period + beta * (exp_market - rf_period)
    expected_return_annual = (1 + expected_return_period) ** periods_per_year - 1
    return {
        'beta': float(beta),
        'alpha_period': float(alpha),
        'expected_return_annual': float(expected_return_annual),
        'rf_period': float(rf_period)
    }


def annualize_return(mean_period_return, periods_per_year):
    return (1 + mean_period_return) ** periods_per_year - 1


def annualize_vol(std_period_return, periods_per_year):
    return std_period_return * np.sqrt(periods_per_year)


def rolling_beta(asset_returns, market_returns, window):
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var


def simulate_portfolios(returns_df, periods_per_year, n_portfolios=5000):
    mean_period = returns_df.mean()
    cov_period = returns_df.cov()
    assets = returns_df.columns.tolist()
    results = []
    for _ in range(n_portfolios):
        w = np.random.random(len(assets))
        w /= np.sum(w)
        port_mean = np.dot(w, mean_period)
        port_var = np.dot(w.T, np.dot(cov_period, w))
        port_return_annual = annualize_return(port_mean, periods_per_year)
        port_vol_annual = np.sqrt(port_var) * np.sqrt(periods_per_year)
        sharpe = (port_return_annual - risk_free / 100.0) / port_vol_annual if port_vol_annual else np.nan
        results.append({
            'weights': w,
            'return_annual': port_return_annual,
            'vol_annual': port_vol_annual,
            'sharpe': sharpe
        })
    return pd.DataFrame([
        {'return_annual': r['return_annual'], 'vol_annual': r['vol_annual'], 'sharpe': r['sharpe'],
         **{f'w_{i}': r['weights'][i] for i in range(len(assets))}} for r in results
    ])

# ========================= MAIN APP ========================= #

if tickers_input.strip() == "":
    st.warning("Please provide at least one NSE ticker (e.g. RELIANCE.NS).")
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
all_tickers = list(dict.fromkeys(tickers + [benchmark.strip().upper()]))

with st.spinner("Fetching price data..."):
    price = fetch_price_data(all_tickers, start_date, end_date)
    if price.empty:
        st.error("No price data returned. Check tickers or date range.")
        st.stop()

periods_per_year = 252 if freq == "Daily" else 12 if freq == "Monthly" else 1
returns = compute_returns(price, freq)
market_col = benchmark.strip().upper()

if market_col not in returns.columns:
    st.error(f"Benchmark {market_col} not found in returns.")
    st.stop()

# === Compute CAPM Metrics ===
detailed = []
for asset in tickers:
    if asset not in returns.columns:
        continue
    res = compute_capm(returns[asset].values, returns[market_col].values, risk_free / 100.0, periods_per_year)
    vol_annual = annualize_vol(returns[asset].std(), periods_per_year)
    mean_period = returns[asset].mean()
    mean_return_annual = annualize_return(mean_period, periods_per_year)
    sharpe = (mean_return_annual - risk_free / 100.0) / vol_annual if vol_annual else np.nan
    treynor = (mean_return_annual - risk_free / 100.0) / res['beta'] if res['beta'] else np.nan
    risk_factor = vol_annual * res['beta'] if not np.isnan(res['beta']) else np.nan

    detailed.append({
        'Asset': asset,
        'Beta': round(res['beta'], 3),
        'Expected Return (Annual %)': round(res['expected_return_annual'] * 100, 2),
        'Volatility (Annual %)': round(vol_annual * 100, 2),
        'Risk Factor (%)': round(risk_factor * 100, 2) if not np.isnan(risk_factor) else np.nan,
        'Sharpe': round(sharpe, 3),
        'Treynor': round(treynor, 3)
    })

df_detailed = pd.DataFrame(detailed).set_index('Asset')
market_return_annual = annualize_return(returns[market_col].mean(), periods_per_year)
market_risk_annual = annualize_vol(returns[market_col].std(), periods_per_year)

# --- Tabs ---
tabs = st.tabs(["Overview", "CAPM Analysis", "Risk & Ratios", "Portfolio", "Forecasting", "Sentiment & Insights"])

# ---------------- Overview ----------------
with tabs[0]:
    st.header("Overview")
    fig = go.Figure()
    for col in price.columns:
        fig.add_trace(go.Scatter(x=price.index, y=price[col], name=col))
    fig.update_layout(height=420, xaxis_title='Date', yaxis_title='Price (INR)')
    st.plotly_chart(fig, use_container_width=True)

# ---------------- CAPM Analysis ----------------
with tabs[1]:
    st.header("CAPM Analysis")
    if df_detailed.empty:
        st.info("No CAPM data available.")
    else:
        st.dataframe(df_detailed)
        fig = px.bar(df_detailed.reset_index(), x='Asset', y='Expected Return (Annual %)',
                     color='Beta', title="Expected Returns vs Beta")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Risk & Ratios ----------------
with tabs[2]:
    st.header("Risk & Ratios")
    if df_detailed.empty:
        st.info("No risk data available.")
    else:
        st.dataframe(df_detailed[['Beta', 'Volatility (Annual %)', 'Sharpe', 'Treynor']])
        window = st.slider("Rolling window (periods)", 3, 252, 60)
        rb_fig = go.Figure()
        for asset in tickers:
            if asset in returns.columns:
                rb = rolling_beta(returns[asset], returns[market_col], window)
                rb_fig.add_trace(go.Scatter(x=rb.index, y=rb, name=asset))
        rb_fig.update_layout(height=420, yaxis_title="Rolling Beta")
        st.plotly_chart(rb_fig, use_container_width=True)

# ---------------- Portfolio ----------------
with tabs[3]:
    st.header("Portfolio Optimization (Monte Carlo)")
    assets_for_portfolio = st.multiselect("Choose assets for portfolio", options=tickers, default=tickers[:min(len(tickers), 4)])
    n_sims = st.slider("Number of simulations", 1000, 20000, 5000, 500)
    if len(assets_for_portfolio) < 2:
        st.info("Select at least two assets.")
    else:
        returns_sub = returns[assets_for_portfolio]
        with st.spinner("Running simulations..."):
            sim_df = simulate_portfolios(returns_sub, periods_per_year, n_portfolios=n_sims)
        fig_pf = px.scatter(sim_df, x='vol_annual', y='return_annual', color='sharpe',
                            title='Simulated Portfolios', labels={'vol_annual': 'Volatility', 'return_annual': 'Return'})
        st.plotly_chart(fig_pf, use_container_width=True)

# ---------------- Forecasting ----------------
with tabs[4]:
    st.header("Forecasting (Simple Moving Average)")
    asset_forecast = st.selectbox("Choose asset to forecast", options=all_tickers, index=0)
    periods_ahead = st.slider("Forecast periods ahead", 1, 52, 12)
    series = price[asset_forecast].dropna()
    if not series.empty:
        window_ma = st.slider("Moving average window", 5, 200, 20)
        ma = series.rolling(window_ma).mean()
        last_ma = ma.dropna().iloc[-1] if not ma.dropna().empty else series.iloc[-1]
        future_index = pd.date_range(start=series.index[-1], periods=periods_ahead+1, inclusive='right')
        forecast = pd.Series([last_ma] * len(future_index), index=future_index)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=series.index, y=series, name='Historical'))
        fig_fc.add_trace(go.Scatter(x=ma.index, y=ma, name=f'MA({window_ma})'))
        fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecast (Naive MA)'))
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.info("No data for forecasting.")

# ---------------- Sentiment ----------------
with tabs[5]:
    st.header("Sentiment & Insights (Demo)")
    corr = returns.corr()
    st.subheader("Return Correlation Matrix")
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Return Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.markdown("Aakash Singh © 2025 · [GitHub]")

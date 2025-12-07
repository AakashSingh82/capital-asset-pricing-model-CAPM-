# capm.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# ---------- Page config ----------
st.set_page_config(
    layout="wide",
    page_title="FinVisionX (CAPM – Indian Market)"
)

# ---------- Sidebar (inputs) ----------
with st.sidebar:
    st.title("FinVisionX Controls")
    tickers_input = st.text_input(
        "Asset tickers (comma separated)",
        value="RELIANCE.NS, TCS.NS, INFY.NS"
    )
    benchmark = st.text_input("Benchmark ticker", value="^NSEI")
    start_date = st.date_input(
        "Start date",
        value=pd.to_datetime("2019-01-01").date()
    )
    end_date = st.date_input("End date", value=date.today())
    freq = st.selectbox(
        "Return frequency",
        ["Daily", "Monthly", "Yearly"],
        index=0
    )
    risk_free = st.number_input(
        "Risk-free rate (annual %)",
        min_value=0.0,
        value=7.0,
        step=0.1
    )
    show_regression = st.checkbox(
        "Show regression line on beta scatter",
        value=True
    )
    sims = st.slider(
        "Monte Carlo simulations (Portfolio)",
        1000, 20000, 5000, step=500
    )
    st.markdown("---")
    st.write("Tips:")
    st.write("- Use `.NS` suffix for NSE tickers (e.g. RELIANCE.NS).")
    st.write("- Choose 2–6 tickers for portfolio optimisation.")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_prices(tickers, start, end):
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by='ticker',
        progress=False
    )
    price = pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                price[t] = raw[(t, "Adj Close")]
            except Exception:
                price[t] = raw[(t, "Close")]
    else:
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        price[tickers[0]] = raw[col]
    price.index = pd.to_datetime(price.index)
    price = price.sort_index().dropna(how='all')
    return price


def compute_returns(price_df, freq):
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    if freq == "Daily":
        return price_df.pct_change().dropna()
    elif freq == "Monthly":
        return price_df.resample("M").last().pct_change().dropna()
    else:  # Yearly
        return price_df.resample("Y").last().pct_change().dropna()


# Regression-based beta (LinearRegression)
def regression_beta(asset_ret, market_ret):
    df = pd.DataFrame({"asset": asset_ret, "market": market_ret}).dropna()
    if df.shape[0] < 2:
        return np.nan, np.nan, np.nan
    X = df["market"].values.reshape(-1, 1)
    y = df["asset"].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return float(model.coef_[0]), float(model.intercept_), float(r2)


# RandomForest-based return forecasting using lag features
def rf_forecast(returns_series, periods_ahead=12, lags=3, n_estimators=200):
    df = pd.DataFrame({"ret": returns_series}).copy()
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["ret"].shift(lag)
    df.dropna(inplace=True)
    if df.shape[0] < 10:
        return [], None, None  # not enough data

    X = df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y = df["ret"].values
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X, y)

    last_lags = list(df[[f"lag_{i}" for i in range(1, lags + 1)]]
                     .iloc[-1].values)
    preds = []
    all_tree_preds = []

    for _ in range(periods_ahead):
        x_in = np.array(last_lags).reshape(1, -1)
        trees_pred = np.array([t.predict(x_in)[0]
                               for t in model.estimators_])
        pred_point = trees_pred.mean()
        preds.append(pred_point)
        all_tree_preds.append(trees_pred)
        last_lags = [pred_point] + last_lags[:-1]

    lower = [np.percentile(all_tree_preds[i], 10)
             for i in range(len(all_tree_preds))]
    upper = [np.percentile(all_tree_preds[i], 90)
             for i in range(len(all_tree_preds))]

    return preds, (lower, upper), model


def rolling_beta(asset_returns, market_returns, window):
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return (cov / var).dropna()


def simulate_portfolios(
    returns_df,
    periods_per_year,
    n_portfolios=5000,
    rf_rate=0.07
):
    mean_period = returns_df.mean()
    cov_period = returns_df.cov()
    assets = returns_df.columns.tolist()
    results = []

    for _ in range(n_portfolios):
        w = np.random.random(len(assets))
        w /= np.sum(w)
        port_mean = np.dot(w, mean_period)
        port_var = np.dot(w.T, np.dot(cov_period, w))
        port_return_annual = (1 + port_mean) ** periods_per_year - 1
        port_vol_annual = np.sqrt(port_var) * np.sqrt(periods_per_year)
        sharpe = (
            (port_return_annual - rf_rate) / port_vol_annual
            if port_vol_annual != 0 else np.nan
        )
        results.append(
            {
                "return": port_return_annual,
                "vol": port_vol_annual,
                "sharpe": sharpe,
                "weights": w,
            }
        )

    df = pd.DataFrame([
        {
            "return": r["return"],
            "vol": r["vol"],
            "sharpe": r["sharpe"],
            **{f"w_{i}": float(r["weights"][i])
               for i in range(len(assets))}
        }
        for r in results
    ])
    return df


# Price indicators: MA, RSI, Bollinger
def add_technical_indicators(price_series):
    df = pd.DataFrame({"price": price_series}).dropna()
    df["ma50"] = df["price"].rolling(50).mean()
    df["ma200"] = df["price"].rolling(200).mean()
    # Bollinger Bands (20,2)
    df["ma20"] = df["price"].rolling(20).mean()
    df["bb_std"] = df["price"].rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * df["bb_std"]
    df["bb_lower"] = df["ma20"] - 2 * df["bb_std"]
    # RSI(14)
    delta = df["price"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    rs = roll_up / roll_down
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def cluster_heatmap_order(corr):
    # convert correlation to distance
    dist = 1 - corr.abs()
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist.values, 0)
    linkage = hierarchy.linkage(
        squareform(dist.values),
        method='average'
    )
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    return corr.columns[dendro['leaves']]


# ---------- Main ----------
st.title("FinVisionX (CAPM – Indian Market)")

if tickers_input.strip() == "":
    st.warning("Please enter at least one ticker (e.g. RELIANCE.NS).")
    st.stop()

tickers = [t.strip().upper()
           for t in tickers_input.split(",") if t.strip()]
all_tickers = list(dict.fromkeys(tickers + [benchmark.strip().upper()]))

with st.spinner("Fetching price data..."):
    price = fetch_prices(all_tickers, start_date, end_date)

if price.empty:
    st.error("No price data found. Check tickers / date range.")
    st.stop()

# ---------- Price & Technical Indicators (Overview) ----------
st.subheader("Price Chart & Technical Indicators")

col1, col2 = st.columns([3, 1])

with col1:
    asset_choice = st.selectbox(
        "Select asset for indicator view",
        all_tickers,
        index=0
    )
    series = price[asset_choice].dropna()
    ind = add_technical_indicators(series)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        name=f"{asset_choice} Price"
    ))
    # Moving averages
    fig.add_trace(go.Scatter(
        x=ind.index,
        y=ind["ma50"],
        name="MA50",
        line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=ind.index,
        y=ind["ma200"],
        name="MA200",
        line=dict(dash="dot")
    ))
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=ind.index,
        y=ind["bb_upper"],
        name="Bollinger Upper",
        line=dict(color="rgba(255,0,0,0.2)")
    ))
    fig.add_trace(go.Scatter(
        x=ind.index,
        y=ind["bb_lower"],
        name="Bollinger Lower",
        line=dict(color="rgba(0,0,255,0.2)"),
        fill='tonexty'
    ))
    fig.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Latest Price", f"{series.iloc[-1]:.2f}")
    ma50_val = ind["ma50"].iloc[-1]
    ma200_val = ind["ma200"].iloc[-1]
    rsi_val = ind["rsi_14"].iloc[-1]
    st.metric(
        "MA50",
        f"{ma50_val:.2f}" if not np.isnan(ma50_val) else "n/a"
    )
    st.metric(
        "MA200",
        f"{ma200_val:.2f}" if not np.isnan(ma200_val) else "n/a"
    )
    st.metric(
        "RSI(14)",
        f"{rsi_val:.2f}" if not np.isnan(rsi_val) else "n/a"
    )

# ---------- Returns ----------
returns = compute_returns(price, freq)
periods_per_year = 252 if freq == "Daily" else 12 if freq == "Monthly" else 1
market = benchmark.strip().upper()
if market not in returns.columns:
    st.error(
        f"Benchmark {market} not in returns. Make sure the benchmark is correct and has data."
    )
    st.stop()

# ---------- Advanced CAPM Analysis ----------
st.divider()
st.subheader("Advanced CAPM Analysis")

beta_rows = []
scatter_fig = go.Figure()

for asset in tickers:
    if asset not in returns.columns:
        continue
    beta, intercept, r2 = regression_beta(
        returns[asset].values,
        returns[market].values
    )
    beta_rows.append(
        {
            "Asset": asset,
            "Regression Beta": round(beta, 4),
            "Alpha": round(intercept, 6),
            "R² (Fit Quality)": round(r2, 4),
        }
    )

    df_sc = pd.DataFrame(
        {"asset": returns[asset], "market": returns[market]}
    ).dropna()
    scatter_fig.add_trace(go.Scatter(
        x=df_sc["market"],
        y=df_sc["asset"],
        mode="markers",
        name=asset,
        opacity=0.6
    ))

    if show_regression and not np.isnan(beta):
        X = df_sc["market"].values.reshape(-1, 1)
        y_pred = (beta * X + intercept).flatten()
        scatter_fig.add_trace(go.Scatter(
            x=df_sc["market"],
            y=y_pred,
            name=f"{asset} regression fit",
            mode="lines",
            opacity=0.9
        ))

scatter_fig.update_layout(
    title="Asset vs Market Returns with Regression Lines",
    xaxis_title=f"{market} returns",
    yaxis_title="Asset returns"
)
st.plotly_chart(scatter_fig, use_container_width=True)
st.table(pd.DataFrame(beta_rows).set_index("Asset"))

# ---------- Rolling Metrics ----------
st.divider()
st.subheader("Risk & Rolling Metrics")

col_rb1, col_rb2 = st.columns(2)
window = st.slider(
    "Rolling window (periods)",
    min_value=10,
    max_value=252,
    value=60
)

with col_rb1:
    rb_fig = go.Figure()
    for asset in tickers:
        if asset in returns.columns:
            rb = rolling_beta(returns[asset], returns[market], window)
            rb_fig.add_trace(go.Scatter(
                x=rb.index,
                y=rb.values,
                name=asset
            ))
    rb_fig.update_layout(
        title=f"Rolling Beta (window = {window})",
        yaxis_title="Beta"
    )
    st.plotly_chart(rb_fig, use_container_width=True)

with col_rb2:
    vol_fig = go.Figure()
    for asset in tickers:
        if asset in returns.columns:
            vol = returns[asset].rolling(window).std() * np.sqrt(periods_per_year)
            vol_fig.add_trace(go.Scatter(
                x=vol.index,
                y=vol.values,
                name=asset
            ))
    vol_fig.update_layout(
        title=f"Rolling Annualised Volatility (window = {window})",
        yaxis_title="Volatility"
    )
    st.plotly_chart(vol_fig, use_container_width=True)

# ---------- Portfolio Optimisation ----------
st.divider()
st.subheader("Efficient Frontier & Portfolio Optimisation")

assets_for_pf = st.multiselect(
    "Choose assets for portfolio simulation",
    options=tickers,
    default=tickers[:min(4, len(tickers))]
)

if len(assets_for_pf) < 2:
    st.info("Select at least 2 assets to run portfolio optimisation.")
else:
    returns_sub = returns[assets_for_pf].dropna()
    sim_df = simulate_portfolios(
        returns_sub,
        periods_per_year,
        n_portfolios=sims,
        rf_rate=risk_free / 100.0
    )

    best_idx = sim_df["sharpe"].idxmax()
    best = sim_df.loc[best_idx]

    fig_pf = px.scatter(
        sim_df,
        x="vol",
        y="return",
        color="sharpe",
        hover_data=[c for c in sim_df.columns if c.startswith("w_")],
        title="Simulated Portfolios (Risk–Return Cloud)",
        labels={
            "vol": "Volatility (annual)",
            "return": "Return (annual)"
        }
    )
    fig_pf.add_trace(go.Scatter(
        x=[best["vol"]],
        y=[best["return"]],
        mode="markers+text",
        marker=dict(size=15, color="gold"),
        text=["Max Sharpe"],
        textposition="top center",
        name="Max Sharpe Portfolio"
    ))
    st.plotly_chart(fig_pf, use_container_width=True)

    weights = [best[f"w_{i}"] for i in range(len(assets_for_pf))]
    w_df = pd.DataFrame(
        {"Asset": assets_for_pf, "Weight": np.round(weights, 4)}
    )
    st.write("Max Sharpe Portfolio Weights:")
    st.dataframe(w_df.set_index("Asset"))

# ---------- Return Forecasting ----------
st.divider()
st.subheader("Return Forecasting")

choose = st.selectbox(
    "Choose asset for return forecasting",
    options=tickers,
    index=0
)
periods_ahead = st.slider(
    "Forecast horizon (steps)",
    3, 36, 12
)
lags = st.slider(
    "Lag features (for model input)",
    1, 8, 3
)

with st.spinner("Training model and generating forecast..."):
    preds, ci, rf_model = rf_forecast(
        returns[choose].values,
        periods_ahead=periods_ahead,
        lags=lags,
        n_estimators=200
    )

if preds == []:
    st.info("Not enough data to run forecasting for this asset.")
else:
    hist_values = (
        returns[choose].values[-200:]
        if len(returns[choose].values) > 200
        else returns[choose].values
    )

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(
        y=hist_values,
        name="Recent historical returns"
    ))

    last_idx = returns[choose].index[-1]
    if freq == "Daily":
        freq_delta = timedelta(days=1)
    elif freq == "Monthly":
        freq_delta = timedelta(days=30)
    else:  # Yearly
        freq_delta = timedelta(days=365)

    forecast_idx = [
        last_idx + (i + 1) * freq_delta for i in range(len(preds))
    ]

    fig_f.add_trace(go.Scatter(
        x=forecast_idx,
        y=preds,
        name="Model forecast",
        line=dict(color="firebrick")
    ))

    if ci:
        lower, upper = ci
        fig_f.add_trace(go.Scatter(
            x=forecast_idx,
            y=upper,
            line=dict(color='lightgrey'),
            name="Upper band",
            showlegend=False
        ))
        fig_f.add_trace(go.Scatter(
            x=forecast_idx,
            y=lower,
            line=dict(color='lightgrey'),
            name="Lower band",
            fill='tonexty',
            fillcolor='rgba(200,200,200,0.2)',
            showlegend=False
        ))

    fig_f.update_layout(
        title=f"Return Forecast for {choose}",
        yaxis_title="Return"
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # ---------- Feature Analysis ----------
    if rf_model is not None:
        st.subheader("Feature Analysis")
        fi = rf_model.feature_importances_
        fi_df = pd.DataFrame(
            {
                "Feature": [f"lag_{i}" for i in range(1, lags + 1)],
                "Importance": fi,
            }
        )
        fi_df = fi_df.sort_values("Importance", ascending=False)
        st.bar_chart(fi_df.set_index("Feature"))

# ---------- Correlation Matrix ----------
st.divider()
st.subheader("Correlation Matrix")

corr = returns.corr()

try:
    order = cluster_heatmap_order(corr)
    corr_clustered = corr.loc[order, order]
    fig_corr = px.imshow(
        corr_clustered,
        text_auto=True,
        aspect="auto",
        title="Clustered Return Correlation"
    )
except Exception:
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Return Correlation"
    )

st.plotly_chart(fig_corr, use_container_width=True)

# ---------- Downloads ----------
st.divider()
st.subheader("Download Data")

col_d1, col_d2 = st.columns(2)
with col_d1:
    price_csv = price.to_csv().encode('utf-8')
    st.download_button(
        "Download Price Data (CSV)",
        data=price_csv,
        file_name="prices_finvisionx.csv",
        mime="text/csv"
    )
with col_d2:
    ret_csv = returns.to_csv().encode('utf-8')
    st.download_button(
        "Download Returns Data (CSV)",
        data=ret_csv,
        file_name="returns_finvisionx.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Aakash Singh © 2025 · FinVisionX")


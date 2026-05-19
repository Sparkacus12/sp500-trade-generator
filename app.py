import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import shapiro, linregress

st.set_page_config(page_title="S&P 500 Trade Generator", layout="wide")

st.title("S&P 500 Trade Generator")

LOOKBACK = 30
P_THRESHOLD = 0.05

@st.cache_data(ttl=60 * 60 * 12)
def get_sp500():
    import requests
    from io import StringIO

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    html = requests.get(url, headers=headers, timeout=20).text
    table = pd.read_html(StringIO(html))[0]

    table["Ticker"] = table["Symbol"].str.replace(".", "-", regex=False)
    return table[["Ticker", "Security", "GICS Sector"]]

@st.cache_data(ttl=60 * 60 * 6)
def get_prices(tickers):
    data = yf.download(
        tickers,
        period="90d",
        auto_adjust=True,
        progress=False,
        threads=True
    )
    return data["Close"].dropna(axis=1, how="all").ffill()

def trend_score(prices):
    y = np.log(prices.values)
    x = np.arange(len(y))
    return linregress(x, y).slope * 100

def analyse_stock(ticker, prices):
    s = prices[ticker].dropna()
    if len(s) < LOOKBACK + 1:
        return None

    window = s.iloc[-LOOKBACK:]
    returns = window.pct_change().dropna()

    if len(returns) < LOOKBACK - 1:
        return None

    pval = shapiro(returns).pvalue
    trend = trend_score(window)
    ret_30d = window.iloc[-1] / window.iloc[0] - 1

    return {
        "Ticker": ticker,
        "Trend score": trend,
        "30d return": ret_30d,
        "Normality p-value": pval,
        "Pass normality": pval > P_THRESHOLD,
    }

def downside_break(ticker, prices):
    s = prices[ticker].dropna()
    if len(s) < LOOKBACK + 2:
        return None

    prior = s.iloc[-LOOKBACK-1:-1]
    current = s.iloc[-LOOKBACK:]

    prior_returns = prior.pct_change().dropna()
    current_returns = current.pct_change().dropna()

    prior_p = shapiro(prior_returns).pvalue
    current_p = shapiro(current_returns).pvalue
    prior_trend = trend_score(prior)
    last_move = s.iloc[-1] / s.iloc[-2] - 1

    if prior_trend > 0 and prior_p > P_THRESHOLD and current_p <= P_THRESHOLD and last_move < 0:
        return {
            "Ticker": ticker,
            "t-1 trend score": prior_trend,
            "Last-day move": last_move,
            "t-1 p-value": prior_p,
            "Current p-value": current_p,
        }
    return None

def upside_break(ticker, prices):
    s = prices[ticker].dropna()
    if len(s) < LOOKBACK + 2:
        return None

    prior = s.iloc[-LOOKBACK-1:-1]
    current = s.iloc[-LOOKBACK:]

    prior_returns = prior.pct_change().dropna()
    current_returns = current.pct_change().dropna()

    prior_p = shapiro(prior_returns).pvalue
    current_p = shapiro(current_returns).pvalue
    prior_trend = trend_score(prior)
    last_move = s.iloc[-1] / s.iloc[-2] - 1

    if prior_trend < 0 and prior_p > P_THRESHOLD and current_p <= P_THRESHOLD and last_move > 0:
        return {
            "Ticker": ticker,
            "t-1 trend score": prior_trend,
            "Last-day move": last_move,
            "t-1 p-value": prior_p,
            "Current p-value": current_p,
        }
    return None

with st.spinner("Loading S&P 500 and price data..."):
    sp500 = get_sp500()
    tickers = sp500["Ticker"].tolist()
    prices = get_prices(tickers)

available = [t for t in tickers if t in prices.columns]

rows = []
down_breaks = []
up_breaks = []

for ticker in available:
    r = analyse_stock(ticker, prices)
    if r:
        rows.append(r)

    d = downside_break(ticker, prices)
    if d:
        down_breaks.append(d)

    u = upside_break(ticker, prices)
    if u:
        up_breaks.append(u)

df = pd.DataFrame(rows).merge(sp500, on="Ticker", how="left")

passed = df[df["Pass normality"]]

buys = (
    passed[passed["Trend score"] > 0]
    .sort_values("Trend score", ascending=False)
    .head(3)
)

sells = (
    passed[passed["Trend score"] < 0]
    .sort_values("Trend score", ascending=True)
    .head(3)
)

st.subheader("Buy recommendations")
st.dataframe(
    buys[["Ticker", "Security", "GICS Sector", "Trend score", "30d return", "Normality p-value"]],
    use_container_width=True
)

st.subheader("Sell recommendations")
st.dataframe(
    sells[["Ticker", "Security", "GICS Sector", "Trend score", "30d return", "Normality p-value"]],
    use_container_width=True
)

st.subheader("Possible sells: positive trend broken by downside move")
if down_breaks:
    ddf = pd.DataFrame(down_breaks).merge(sp500, on="Ticker", how="left")
    st.dataframe(ddf, use_container_width=True)
else:
    st.write("No downside trend-break candidates today.")

st.subheader("Possible buys: negative trend broken by upside move")
if up_breaks:
    udf = pd.DataFrame(up_breaks).merge(sp500, on="Ticker", how="left")
    st.dataframe(udf, use_container_width=True)
else:
    st.write("No upside trend-break candidates today.")

st.caption(
    "Systematic screen only, not investment advice. "
    "Uses Yahoo Finance adjusted daily prices via yfinance. "
    "Normality test is Shapiro-Wilk on last 30 daily returns."
)
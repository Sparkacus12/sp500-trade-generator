import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import shapiro, linregress

st.set_page_config(page_title="S&P 500 Trade Generator", layout="wide")

st.title("S&P 500 Trade Generator")

LOOKBACK = 30
P_THRESHOLD = 0.10
PRICE_PERIOD = "3y"

st.caption(
    "Systematic screen only, not investment advice. "
    "Uses Yahoo Finance adjusted daily prices via yfinance. "
    "Normality test is Shapiro-Wilk on rolling 30-day returns."
)

@st.cache_data(ttl=60 * 60 * 12)
def get_sp500():
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
        period=PRICE_PERIOD,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return data["Close"].dropna(axis=1, how="all").ffill()

def trend_score(series):
    y = np.log(series.values)
    x = np.arange(len(y))
    return linregress(x, y).slope * 100

def normality_pvalue(series):
    returns = series.pct_change().dropna()
    if len(returns) < LOOKBACK - 1:
        return np.nan
    return shapiro(returns).pvalue

MACRO_TICKERS = {
    "Market": "SPY",
    "Credit": "HYG",
    "Rates": "TLT",
    "Dollar": "UUP",
    "Oil": "USO",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

@st.cache_data(ttl=60 * 60 * 6)
def get_macro_prices():
    data = yf.download(
        list(MACRO_TICKERS.values()),
        period=PRICE_PERIOD,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return data["Close"].dropna(axis=1, how="all").ffill()

def pct_return(prices, ticker, days=30):
    try:
        s = prices[ticker].dropna()
        if len(s) < days + 1:
            return 0
        return s.iloc[-1] / s.iloc[-days] - 1
    except Exception:
        return 0

def macro_score(sector, ticker, prices, macro_prices):
    sector_etf = MACRO_TICKERS.get(sector)

    sector_mom = pct_return(macro_prices, sector_etf)
    market_mom = pct_return(macro_prices, "SPY")
    credit_mom = pct_return(macro_prices, "HYG")
    rates_mom = pct_return(macro_prices, "TLT")
    dollar_mom = pct_return(macro_prices, "UUP")
    oil_mom = pct_return(macro_prices, "USO")

    score = (
        45 * sector_mom
        + 25 * market_mom
        + 20 * credit_mom
        - 10 * rates_mom
        - 10 * dollar_mom
    )

    if sector == "Energy":
        score += 25 * oil_mom

    try:
        stock_ret = pct_return(prices, ticker)
        sector_ret = pct_return(macro_prices, sector_etf)
        relative_strength = (stock_ret - sector_ret) * 100
    except Exception:
        relative_strength = 0

    return score + relative_strength

with st.spinner("Loading S&P 500, prices and macro proxies..."):
    sp500 = get_sp500()
    tickers = sp500["Ticker"].tolist()
    prices = get_prices(tickers)
    macro_prices = get_macro_prices()

available = [t for t in tickers if t in prices.columns]

rows = []

for ticker in available:
    s = prices[ticker].dropna()
    if len(s) < LOOKBACK + 1:
        continue

    window = s.iloc[-LOOKBACK:]
    pval = normality_pvalue(window)
    if np.isnan(pval):
        continue

    trend = trend_score(window)
    ret_30d = window.iloc[-1] / window.iloc[0] - 1

    rows.append({
        "Ticker": ticker,
        "Trend score": trend,
        "30d return": ret_30d,
        "Normality p-value": pval,
        "Pass normality": pval > P_THRESHOLD,
    })

df = pd.DataFrame(rows).merge(sp500, on="Ticker", how="left")

df["Macro score"] = df.apply(
    lambda r: macro_score(r["GICS Sector"], r["Ticker"], prices, macro_prices),
    axis=1,
)

df["Macro signal"] = np.where(
    df["Macro score"] > 2,
    "Positive macro overlay",
    np.where(df["Macro score"] < -2, "Negative macro overlay", "Neutral"),
)

passed = df[df["Pass normality"]]

buys = (
    passed[passed["Trend score"] > 0]
    .sort_values(["Trend score", "Macro score"], ascending=[False, False])
    .head(3)
)

sells = (
    passed[passed["Trend score"] < 0]
    .sort_values(["Trend score", "Macro score"], ascending=[True, True])
    .head(3)
)

st.subheader("Strategy 1: Core daily recommendations")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Buys")
    st.dataframe(
        buys[[
            "Ticker", "Security", "GICS Sector", "Trend score",
            "30d return", "Normality p-value", "Macro score", "Macro signal"
        ]],
        use_container_width=True,
    )

with c2:
    st.markdown("### Sells")
    st.dataframe(
        sells[[
            "Ticker", "Security", "GICS Sector", "Trend score",
            "30d return", "Normality p-value", "Macro score", "Macro signal"
        ]],
        use_container_width=True,
    )

st.subheader("Trend-break watchlist")

down_breaks = []
up_breaks = []

for ticker in available:
    s = prices[ticker].dropna()
    if len(s) < LOOKBACK + 2:
        continue

    prior = s.iloc[-LOOKBACK - 1:-1]
    current = s.iloc[-LOOKBACK:]

    prior_p = normality_pvalue(prior)
    current_p = normality_pvalue(current)
    prior_trend = trend_score(prior)
    last_move = s.iloc[-1] / s.iloc[-2] - 1

    if prior_trend > 0 and prior_p > P_THRESHOLD and current_p <= P_THRESHOLD and last_move < 0:
        down_breaks.append({
            "Ticker": ticker,
            "t-1 trend score": prior_trend,
            "Last-day move": last_move,
            "t-1 p-value": prior_p,
            "Current p-value": current_p,
        })

    if prior_trend < 0 and prior_p > P_THRESHOLD and current_p <= P_THRESHOLD and last_move > 0:
        up_breaks.append({
            "Ticker": ticker,
            "t-1 trend score": prior_trend,
            "Last-day move": last_move,
            "t-1 p-value": prior_p,
            "Current p-value": current_p,
        })

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Possible sells: positive trend broken by downside move")
    if down_breaks:
        ddf = pd.DataFrame(down_breaks).merge(sp500, on="Ticker", how="left")
        st.dataframe(ddf, use_container_width=True)
    else:
        st.write("No downside break candidates today.")

with c2:
    st.markdown("### Possible buys: negative trend broken by upside move")
    if up_breaks:
        udf = pd.DataFrame(up_breaks).merge(sp500, on="Ticker", how="left")
        st.dataframe(udf, use_container_width=True)
    else:
        st.write("No upside break candidates today.")

st.subheader("Macro-earnings overlay")

macro_table = df[[
    "Ticker", "Security", "GICS Sector", "Macro score", "Macro signal"
]].sort_values("Macro score", ascending=False)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Top positive macro overlays")
    st.dataframe(macro_table.head(10), use_container_width=True)

with c2:
    st.markdown("### Top negative macro overlays")
    st.dataframe(
        macro_table.tail(10).sort_values("Macro score"),
        use_container_width=True,
    )

st.subheader("Backtests")

if not st.button("Run backtests"):
    st.info("Click to run the backtests. This can take a few minutes.")
    st.stop()

@st.cache_data(ttl=60 * 60 * 6, show_spinner=True)
def run_strategy_1_backtest(prices, lookback=30, p_threshold=P_THRESHOLD):
    returns = prices.pct_change()
    results = []

    for i in range(lookback + 1, len(prices) - 1):
        trade_date = prices.index[i + 1]
        candidates = []

        for ticker in prices.columns:
            window = prices[ticker].iloc[i - lookback:i].dropna()
            if len(window) < lookback:
                continue

            pval = normality_pvalue(window)
            if np.isnan(pval) or pval <= p_threshold:
                continue

            trend = trend_score(window)

            candidates.append({
                "Ticker": ticker,
                "Trend": trend,
                "P-value": pval,
            })

        c = pd.DataFrame(candidates)

        if c.empty:
            continue

        longs = c[c["Trend"] > 0].sort_values("Trend", ascending=False).head(3)
        shorts = c[c["Trend"] < 0].sort_values("Trend", ascending=True).head(3)

        if len(longs) < 3 or len(shorts) < 3:
            continue

        next_returns = returns.loc[trade_date]

        long_return = next_returns[longs["Ticker"]].mean()
        short_return = next_returns[shorts["Ticker"]].mean()

        results.append({
            "Date": trade_date,
            "Return": long_return - short_return,
            "Longs": ", ".join(longs["Ticker"]),
            "Shorts": ", ".join(shorts["Ticker"]),
        })

    return pd.DataFrame(results)

@st.cache_data(ttl=60 * 60 * 6, show_spinner=True)
def run_strategy_2_backtest(prices, df, lookback=30, p_threshold=P_THRESHOLD):
    returns = prices.pct_change()
    positions = {}
    results = []
    trade_log = []

    macro_lookup = df.set_index("Ticker")["Macro score"].to_dict()

    for i in range(lookback + 2, len(prices) - 1):
        signal_date = prices.index[i]
        trade_date = prices.index[i + 1]

        for ticker in list(positions.keys()):
            window = prices[ticker].iloc[i - lookback:i].dropna()
            if len(window) < lookback:
                del positions[ticker]
                continue

            pval = normality_pvalue(window)
            trend = trend_score(window)
            side = positions[ticker]
            macro_value = macro_lookup.get(ticker, 0)

            exit_reason = None

            if pval <= p_threshold:
                exit_reason = "Normality broken"
            elif side == "LONG" and trend <= 0:
                exit_reason = "Positive trend broken"
            elif side == "SHORT" and trend >= 0:
                exit_reason = "Negative trend broken"
            elif side == "LONG" and macro_value < 0:
                exit_reason = "Macro no longer supports long"
            elif side == "SHORT" and macro_value > 0:
                exit_reason = "Macro no longer supports short"

            if exit_reason:
                trade_log.append({
                    "Date": signal_date,
                    "Ticker": ticker,
                    "Action": "EXIT",
                    "Side": side,
                    "Reason": exit_reason,
                    "P-value": pval,
                    "Trend": trend,
                    "Macro score": macro_value,
                })
                del positions[ticker]

        new_longs = []
        new_shorts = []

        for ticker in prices.columns:
            if ticker in positions:
                continue

            prior = prices[ticker].iloc[i - lookback - 1:i - 1].dropna()
            current = prices[ticker].iloc[i - lookback:i].dropna()

            if len(prior) < lookback or len(current) < lookback:
                continue

            prior_p = normality_pvalue(prior)
            current_p = normality_pvalue(current)

            if not (prior_p <= p_threshold and current_p > p_threshold):
                continue

            trend = trend_score(current)
            ret_30d = current.iloc[-1] / current.iloc[0] - 1
            macro_value = macro_lookup.get(ticker, 0)

            if trend > 0 and ret_30d > 0 and macro_value > 0:
                new_longs.append({
                    "Ticker": ticker,
                    "Rank score": trend + macro_value,
                    "P-value": current_p,
                    "Trend": trend,
                    "Macro score": macro_value,
                })

            elif trend < 0 and ret_30d < 0 and macro_value < 0:
                new_shorts.append({
                    "Ticker": ticker,
                    "Rank score": abs(trend) + abs(macro_value),
                    "P-value": current_p,
                    "Trend": trend,
                    "Macro score": macro_value,
                })

        if new_longs:
            new_longs = pd.DataFrame(new_longs).sort_values("Rank score", ascending=False).head(3)
            for _, row in new_longs.iterrows():
                positions[row["Ticker"]] = "LONG"
                trade_log.append({
                    "Date": signal_date,
                    "Ticker": row["Ticker"],
                    "Action": "ENTER",
                    "Side": "LONG",
                    "Reason": "Normalised with positive trend, return and macro support",
                    "P-value": row["P-value"],
                    "Trend": row["Trend"],
                    "Macro score": row["Macro score"],
                })

        if new_shorts:
            new_shorts = pd.DataFrame(new_shorts).sort_values("Rank score", ascending=False).head(3)
            for _, row in new_shorts.iterrows():
                positions[row["Ticker"]] = "SHORT"
                trade_log.append({
                    "Date": signal_date,
                    "Ticker": row["Ticker"],
                    "Action": "ENTER",
                    "Side": "SHORT",
                    "Reason": "Normalised with negative trend, return and macro support",
                    "P-value": row["P-value"],
                    "Trend": row["Trend"],
                    "Macro score": row["Macro score"],
                })

        next_returns = returns.loc[trade_date]

        long_tickers = [t for t, side in positions.items() if side == "LONG"]
        short_tickers = [t for t, side in positions.items() if side == "SHORT"]

        long_return = next_returns[long_tickers].mean() if long_tickers else 0
        short_return = next_returns[short_tickers].mean() if short_tickers else 0

        results.append({
            "Date": trade_date,
            "Return": long_return - short_return,
            "Number longs": len(long_tickers),
            "Number shorts": len(short_tickers),
            "Longs": ", ".join(long_tickers),
            "Shorts": ", ".join(short_tickers),
        })

    return pd.DataFrame(results), pd.DataFrame(trade_log)

def show_backtest(name, bt):
    st.markdown(f"### {name}")

    if bt.empty:
        st.write("No results generated.")
        return

    bt = bt.dropna()
    bt["Equity curve"] = (1 + bt["Return"]).cumprod()

    total_return = bt["Equity curve"].iloc[-1] - 1
    ann_return = bt["Equity curve"].iloc[-1] ** (252 / len(bt)) - 1
    ann_vol = bt["Return"].std() * np.sqrt(252)

    if bt["Return"].std() != 0:
        sharpe = bt["Return"].mean() / bt["Return"].std() * np.sqrt(252)
    else:
        sharpe = np.nan

    drawdown = bt["Equity curve"] / bt["Equity curve"].cummax() - 1
    max_drawdown = drawdown.min()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"{total_return:.2%}")
    c2.metric("Annualised return", f"{ann_return:.2%}")
    c3.metric("Sharpe ratio", f"{sharpe:.2f}")
    c4.metric("Max drawdown", f"{max_drawdown:.2%}")

    st.line_chart(bt.set_index("Date")["Equity curve"])
    st.dataframe(bt.tail(20), use_container_width=True)

bt1 = run_strategy_1_backtest(prices)
show_backtest("Strategy 1: Daily top 3 long / bottom 3 short", bt1)

bt2, trades2 = run_strategy_2_backtest(prices, df)
show_backtest("Strategy 2: Ranked normalisation regime shift, hold until break", bt2)

st.markdown("### Strategy 2 trade log")
if trades2.empty:
    st.write("No Strategy 2 trades.")
else:
    st.dataframe(trades2.tail(50), use_container_width=True)
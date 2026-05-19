import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import shapiro, linregress

st.set_page_config(page_title="S&P 500 Trade Generator", layout="wide")

st.title("S&P 500 Trade Generator")

LOOKBACK = 30
P_THRESHOLD = 0.10

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
        period="3Y",
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
st.subheader("Macro-earnings overlay")

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
        period="3y",
        auto_adjust=True,
        progress=False,
        threads=True
    )
    return data["Close"].dropna(axis=1, how="all").ffill()

def macro_return(prices, ticker, days=30):
    if ticker not in prices.columns:
        return 0
    s = prices[ticker].dropna()
    if len(s) < days + 1:
        return 0
    return s.iloc[-1] / s.iloc[-days] - 1

def macro_score(sector, macro_prices):
    sector_etf = MACRO_TICKERS.get(sector)

    sector_mom = macro_return(macro_prices, sector_etf)
    market_mom = macro_return(macro_prices, "SPY")
    credit_mom = macro_return(macro_prices, "HYG")
    rates_mom = macro_return(macro_prices, "TLT")
    dollar_mom = macro_return(macro_prices, "UUP")
    oil_mom = macro_return(macro_prices, "USO")

    score = (
        45 * sector_mom +
        25 * market_mom +
        20 * credit_mom -
        10 * rates_mom -
        10 * dollar_mom
    )

    if sector == "Energy":
        score += 25 * oil_mom

    return score

macro_prices = get_macro_prices()

def stock_relative_strength(ticker, sector, prices, macro_prices):
    try:
        sector_etf = MACRO_TICKERS.get(sector)

        stock_series = prices[ticker].dropna()
        sector_series = macro_prices[sector_etf].dropna()

        stock_ret = stock_series.iloc[-1] / stock_series.iloc[-30] - 1
        sector_ret = sector_series.iloc[-1] / sector_series.iloc[-30] - 1

        return (stock_ret - sector_ret) * 100

    except:
        return 0

macro_scores = []

for _, row in df.iterrows():

    sector_component = macro_score(
        row["GICS Sector"],
        macro_prices
    )

    relative_strength = stock_relative_strength(
        row["Ticker"],
        row["GICS Sector"],
        prices,
        macro_prices
    )

    total_score = sector_component + relative_strength

    macro_scores.append(total_score)

df["Macro score"] = macro_scores

df["Macro signal"] = np.where(
    df["Macro score"] > 2,
    "Positive macro-earnings overlay",
    np.where(
        df["Macro score"] < -2,
        "Negative macro-earnings overlay",
        "Neutral"
    )
)

macro_table = df[
    ["Ticker", "Security", "GICS Sector", "Macro score", "Macro signal"]
].sort_values("Macro score", ascending=False)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top positive macro overlays**")
    st.dataframe(macro_table.head(10), use_container_width=True)

with col2:
    st.markdown("**Top negative macro overlays**")
    st.dataframe(
        macro_table.tail(10).sort_values("Macro score"),
        use_container_width=True
    )
    
    st.subheader("Backtest: daily long/short strategy")

def run_backtest(prices, lookback=30, p_threshold=P_THRESHOLD):
    returns = prices.pct_change()
    results = []

    for i in range(lookback + 1, len(prices) - 1):
        signal_date = prices.index[i]
        trade_date = prices.index[i + 1]

        candidates = []

        for ticker in prices.columns:
            s = prices[ticker].iloc[i - lookback:i].dropna()

            if len(s) < lookback:
                continue

            r = s.pct_change().dropna()

            if len(r) < lookback - 1:
                continue

            try:
                pval = shapiro(r).pvalue
            except:
                continue

            if pval <= p_threshold:
                continue

            trend = trend_score(s)

            candidates.append({
                "Ticker": ticker,
                "Trend": trend,
                "P-value": pval
            })

        c = pd.DataFrame(candidates)

        if c.empty:
            continue

        longs = (
            c[c["Trend"] > 0]
            .sort_values("Trend", ascending=False)
            .head(3)
        )

        shorts = (
            c[c["Trend"] < 0]
            .sort_values("Trend", ascending=True)
            .head(3)
        )

        if len(longs) < 3 or len(shorts) < 3:
            continue

        next_returns = returns.loc[trade_date]

        long_return = next_returns[longs["Ticker"]].mean()
        short_return = next_returns[shorts["Ticker"]].mean()

        portfolio_return = long_return - short_return

        results.append({
            "Date": trade_date,
            "Return": portfolio_return,
            "Longs": ", ".join(longs["Ticker"]),
            "Shorts": ", ".join(shorts["Ticker"])
        })

    return pd.DataFrame(results)

bt = run_backtest(prices)

if bt.empty:
    st.write("No backtest results generated. Try lowering the normality threshold.")
else:
    bt = bt.dropna()
    bt["Equity curve"] = (1 + bt["Return"]).cumprod()

    total_return = bt["Equity curve"].iloc[-1] - 1
    annualised_return = bt["Equity curve"].iloc[-1] ** (252 / len(bt)) - 1
    annualised_vol = bt["Return"].std() * np.sqrt(252)
    sharpe = bt["Return"].mean() / bt["Return"].std() * np.sqrt(252)

    running_max = bt["Equity curve"].cummax()
    drawdown = bt["Equity curve"] / running_max - 1
    max_drawdown = drawdown.min()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total return", f"{total_return:.2%}")
    c2.metric("Annualised return", f"{annualised_return:.2%}")
    c3.metric("Sharpe ratio", f"{sharpe:.2f}")
    c4.metric("Max drawdown", f"{max_drawdown:.2%}")

    st.line_chart(bt.set_index("Date")["Equity curve"])

    st.markdown("**Recent backtest trades**")
    st.dataframe(bt.tail(20), use_container_width=True)
    
    st.subheader("Strategy 2: Normalisation regime-shift strategy")

def run_normalisation_hold_until_break_backtest(prices, lookback=30, p_threshold=P_THRESHOLD):
    returns = prices.pct_change()
    positions = {}
    results = []
    trade_log = []

    for i in range(lookback + 2, len(prices) - 1):
        signal_date = prices.index[i]
        trade_date = prices.index[i + 1]

        daily_position_returns = []

        # 1. Update existing positions and exit if normality breaks
        for ticker in list(positions.keys()):
            window = prices[ticker].iloc[i - lookback:i].dropna()

            if len(window) < lookback:
                del positions[ticker]
                continue

            window_returns = window.pct_change().dropna()

            try:
                pval = shapiro(window_returns).pvalue
            except:
                del positions[ticker]
                continue

            if pval <= p_threshold:
                trade_log.append({
                    "Date": signal_date,
                    "Ticker": ticker,
                    "Action": "EXIT",
                    "Side": positions[ticker],
                    "Reason": "Normality broken",
                    "P-value": pval,
                })
                del positions[ticker]

        # 2. Look for new entries: non-normal yesterday, normal today
        for ticker in prices.columns:
            if ticker in positions:
                continue

            prior_window = prices[ticker].iloc[i - lookback - 1:i - 1].dropna()
            current_window = prices[ticker].iloc[i - lookback:i].dropna()

            if len(prior_window) < lookback or len(current_window) < lookback:
                continue

            prior_returns = prior_window.pct_change().dropna()
            current_returns = current_window.pct_change().dropna()

            try:
                prior_p = shapiro(prior_returns).pvalue
                current_p = shapiro(current_returns).pvalue
            except:
                continue

            if not (prior_p <= p_threshold and current_p > p_threshold):
                continue

            trend = trend_score(current_window)

            if trend > 0:
                positions[ticker] = "LONG"
                trade_log.append({
                    "Date": signal_date,
                    "Ticker": ticker,
                    "Action": "ENTER",
                    "Side": "LONG",
                    "Reason": "Normalised with positive trend",
                    "P-value": current_p,
                })

            elif trend < 0:
                positions[ticker] = "SHORT"
                trade_log.append({
                    "Date": signal_date,
                    "Ticker": ticker,
                    "Action": "ENTER",
                    "Side": "SHORT",
                    "Reason": "Normalised with negative trend",
                    "P-value": current_p,
                })

        # 3. Calculate next-day P&L from active positions
        next_returns = returns.loc[trade_date]

        long_tickers = [t for t, side in positions.items() if side == "LONG"]
        short_tickers = [t for t, side in positions.items() if side == "SHORT"]

        long_return = next_returns[long_tickers].mean() if long_tickers else 0
        short_return = next_returns[short_tickers].mean() if short_tickers else 0

        portfolio_return = long_return - short_return

        results.append({
            "Date": trade_date,
            "Return": portfolio_return,
            "Number longs": len(long_tickers),
            "Number shorts": len(short_tickers),
            "Longs": ", ".join(long_tickers),
            "Shorts": ", ".join(short_tickers),
        })

    return pd.DataFrame(results), pd.DataFrame(trade_log)


shift_bt, shift_trades = run_normalisation_hold_until_break_backtest(prices)

if shift_bt.empty:
    st.write("No Strategy 2 trades generated.")
else:
    shift_bt = shift_bt.dropna()
    shift_bt["Equity curve"] = (1 + shift_bt["Return"]).cumprod()

    shift_total_return = shift_bt["Equity curve"].iloc[-1] - 1
    shift_annualised_return = shift_bt["Equity curve"].iloc[-1] ** (252 / len(shift_bt)) - 1
    shift_annualised_vol = shift_bt["Return"].std() * np.sqrt(252)

    if shift_bt["Return"].std() != 0:
        shift_sharpe = shift_bt["Return"].mean() / shift_bt["Return"].std() * np.sqrt(252)
    else:
        shift_sharpe = np.nan

    shift_running_max = shift_bt["Equity curve"].cummax()
    shift_drawdown = shift_bt["Equity curve"] / shift_running_max - 1
    shift_max_drawdown = shift_drawdown.min()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total return", f"{shift_total_return:.2%}")
    c2.metric("Annualised return", f"{shift_annualised_return:.2%}")
    c3.metric("Sharpe ratio", f"{shift_sharpe:.2f}")
    c4.metric("Max drawdown", f"{shift_max_drawdown:.2%}")

    st.line_chart(shift_bt.set_index("Date")["Equity curve"])

    st.write("Number of Strategy 2 trading days:", len(shift_bt))
    st.write("Average number of longs:", round(shift_bt["Number longs"].mean(), 2))
    st.write("Average number of shorts:", round(shift_bt["Number shorts"].mean(), 2))

    st.markdown("**Recent Strategy 2 positions**")
    st.dataframe(shift_bt.tail(20), use_container_width=True)

    st.markdown("**Recent Strategy 2 trade log**")
    st.dataframe(shift_trades.tail(30), use_container_width=True)





 

            
            
                
        

    c1,
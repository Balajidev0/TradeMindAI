import numpy as np
import pandas as pd
import yfinance as yf


def _normalize_symbol(symbol: str, exchange: str) -> str:
    exchange = exchange.strip().upper()
    normalized = symbol.strip().upper()
    if exchange == "NSE":
        if normalized.startswith("^"):
            return normalized
        if not normalized.endswith(".NS"):
            normalized += ".NS"
    return normalized


def fetch_stock_data(symbol: str, period: str = "1mo", interval: str = "1d", exchange: str = "US") -> pd.DataFrame:
    normalized_symbol = _normalize_symbol(symbol, exchange)
    ticker = yf.Ticker(normalized_symbol)
    data = ticker.history(period=period, interval=interval, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for symbol: {normalized_symbol}")
    return data


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "Close" not in data.columns:
        return data.copy()

    result = data.copy()
    close = result["Close"]
    result["SMA_20"] = calculate_sma(close, 20)
    result["SMA_50"] = calculate_sma(close, 50)
    result["RSI_14"] = calculate_rsi(close, 14)
    if "High" in result.columns and "Low" in result.columns:
        result["ATR_14"] = calculate_atr(result, 14)
    return result


def identify_swing_trade_signal(data: pd.DataFrame) -> dict[str, str | float | None]:
    if data.empty or "Close" not in data.columns:
        return {
            "signal": "none",
            "reason": "Insufficient data",
            "stop_loss": None,
            "target": None,
        }

    latest = data.iloc[-1]
    if any(col not in data.columns for col in ["SMA_20", "SMA_50", "RSI_14"]):
        return {
            "signal": "none",
            "reason": "Missing technical indicators",
            "stop_loss": None,
            "target": None,
        }

    price = float(latest["Close"])
    sma20 = float(latest["SMA_20"])
    sma50 = float(latest["SMA_50"])
    rsi = float(latest["RSI_14"])
    atr = float(latest.get("ATR_14", np.nan)) if "ATR_14" in latest else np.nan

    reasons = []
    signal = "hold"

    if price > sma20 and sma20 > sma50 and 40 <= rsi <= 70:
        signal = "buy"
        reasons.append("Price is above SMA20 and SMA20 above SMA50")
        if rsi < 60:
            reasons.append("RSI is in the bullish range")
    elif price < sma20 and price < sma50 and rsi > 70:
        signal = "sell"
        reasons.append("Price is below both SMA20 and SMA50")
        reasons.append("RSI is overbought")
    elif rsi < 30:
        signal = "buy"
        reasons.append("RSI indicates oversold conditions")
    elif rsi > 70:
        signal = "sell"
        reasons.append("RSI indicates overbought conditions")
    else:
        reasons.append("No strong swing signal found")

    stop_loss = None
    target = None
    if signal in {"buy", "sell"} and not np.isnan(atr):
        buffer = atr * 0.8
        stop_loss = float(price - buffer) if signal == "buy" else float(price + buffer)
        target = float(price + atr * 2) if signal == "buy" else float(price - atr * 2)
        reasons.append(f"ATR-based risk buffer is {buffer:.2f}")

    return {
        "signal": signal,
        "reason": "; ".join(reasons),
        "stop_loss": stop_loss,
        "target": target,
    }


def _trend_summary(latest: float, sma: float | None) -> str:
    if sma is None or np.isnan(sma):
        return "Neutral"
    return "Bullish" if latest >= sma else "Bearish"


def _vix_label(vix: float) -> str:
    if vix < 12:
        return "Low"
    if vix < 18:
        return "Moderate"
    return "High"


def _market_health_score(nifty_change: float, vix: float) -> int:
    score = 50 + int(nifty_change * 50)
    score -= int(max(0, vix - 12) * 2)
    return max(0, min(100, score))


def _sector_leader(symbols: dict[str, list[str]], exchange: str) -> str:
    sector_scores: dict[str, float] = {}
    for sector, tickers in symbols.items():
        values = []
        for ticker in tickers:
            try:
                data = fetch_stock_data(ticker, period="1mo", interval="1d", exchange=exchange)
                if not data.empty and "Close" in data.columns:
                    values.append(float(data["Close"].iloc[-1] / data["Close"].iloc[0] - 1))
            except ValueError:
                continue
        sector_scores[sector] = float(np.nanmean(values)) if values else 0.0

    leader = max(sector_scores.items(), key=lambda item: item[1])
    return f"{leader[0]} ({leader[1] * 100:+.2f}%)" if sector_scores else "N/A"


def scan_top_swing_picks(exchange: str = "NSE", limit: int = 20) -> list[dict[str, str | float]]:
    sample_symbols = [
        "BEL", "HAL", "INFY", "SBIN", "LT", "TCS", "HCLTECH", "RELIANCE", "ICICIBANK", "HDFC",
        "M&M", "MARUTI", "TATAMOTORS", "ONGC", "BPCL", "AXISBANK", "KOTAKBANK", "ITC", "BAJAJ-AUTO", "JSWSTEEL",
    ]
    results = []
    for symbol in sample_symbols:
        ticker = symbol if exchange == "US" else f"{symbol}.NS"
        try:
            data = fetch_stock_data(ticker, period="3mo", interval="1d", exchange=exchange)
            data = add_technical_indicators(data)
            signal = identify_swing_trade_signal(data)
            score = 20.0
            if signal["signal"] == "buy":
                score += 40.0
            elif signal["signal"] == "sell":
                score += 10.0
            score += float(data["RSI_14"].iloc[-1]) if "RSI_14" in data.columns and not pd.isna(data["RSI_14"].iloc[-1]) else 0
            results.append({
                "symbol": symbol,
                "signal": signal["signal"],
                "score": round(min(100, score), 2),
                "reason": signal["reason"],
            })
        except Exception:
            continue
    return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


def generate_trade_recommendation(symbol: str, period: str = "1mo", interval: str = "1d", exchange: str = "US") -> dict[str, str | float]:
    data = fetch_stock_data(symbol, period=period, interval=interval, exchange=exchange)
    data = add_technical_indicators(data)
    signal = identify_swing_trade_signal(data)

    latest = data.iloc[-1]
    price = float(latest["Close"])
    stop_loss = signal["stop_loss"] or 0.0
    target = signal["target"] or price
    probability = 80.0 if signal["signal"] == "buy" else 30.0 if signal["signal"] == "sell" else 50.0
    confidence = "High" if probability >= 75 else "Medium" if probability >= 50 else "Low"
    trade_mind_score = min(100, max(0, score := round(40 + (probability * 0.5), 2)))

    return {
        "Stock": symbol,
        "TradeMind Score": trade_mind_score,
        "Probability": f"{probability:.0f}%",
        "Confidence": confidence,
        "Entry": f"{price:.2f}",
        "Stop Loss": f"{stop_loss:.2f}",
        "Target": f"{target:.2f}",
        "Holding": "7-10 Days",
        "Risk": "Low" if signal["signal"] == "buy" else "Medium" if signal["signal"] == "sell" else "High",
        "Reason": signal["reason"],
    }


def generate_market_dashboard(exchange: str = "NSE") -> str:
    symbol_map = {
        "NSE": {
            "nifty": "^NSEI",
            "bank_nifty": "^NSEBANK",
            "vix": "^INDIAVIX",
            "sample_symbols": [
                "BEL", "HAL", "INFY", "SBIN", "LT", "TCS", "HCLTECH", "RELIANCE", "ICICIBANK",
            ],
            "sector_groups": {
                "IT": ["INFY", "TCS", "HCLTECH"],
                "Banking": ["SBIN", "ICICIBANK", "HDFC"],
                "Auto": ["M&M", "MARUTI", "TATAMOTORS"],
                "Energy": ["RELIANCE", "ONGC", "BPCL"],
            },
        }
    }
    market = symbol_map.get(exchange, symbol_map["NSE"])

    def fetch_value(symbol: str) -> tuple[float | None, float | None]:
        try:
            data = fetch_stock_data(symbol, period="1mo", interval="1d", exchange=exchange)
            close = data["Close"].dropna()
            if len(close) < 2:
                return None, None
            return float(close.iloc[-1]), float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else None
        except Exception:
            return None, None

    nifty_close, nifty_change = fetch_value(market["nifty"])
    bank_close, bank_change = fetch_value(market["bank_nifty"])
    vix_data = fetch_stock_data(market["vix"], period="1mo", interval="1d", exchange=exchange)
    vix_level = float(vix_data["Close"].dropna().iloc[-1]) if not vix_data.empty else np.nan

    if nifty_close is not None and nifty_change is not None:
        health_score = _market_health_score(nifty_change, vix_level)
        nifty_trend = "Bullish" if nifty_change >= 0 else "Bearish"
    else:
        health_score = 50
        nifty_trend = "Neutral"

    if bank_close is not None and bank_change is not None:
        bank_trend = "Bullish" if bank_change >= 0 else "Bearish"
    else:
        bank_trend = "Neutral"

    sector_leader = _sector_leader(market["sector_groups"], exchange)
    fii_activity = "Positive" if nifty_change is not None and nifty_change >= 0 else "Negative"

    picks = []
    for symbol in market["sample_symbols"]:
        try:
            ticker = symbol if exchange == "US" else f"{symbol}.NS"
            data = fetch_stock_data(ticker, period="5d", interval="1d", exchange=exchange)
            close = data["Close"].dropna()
            if len(close) >= 2:
                picks.append((symbol, float(close.iloc[-1] / close.iloc[-2] - 1)))
        except Exception:
            continue

    picks = sorted(picks, key=lambda item: item[1], reverse=True)[:5]
    if not picks:
        picks = [(symbol, 0.0) for symbol in market["sample_symbols"][:5]]

    headlines = [
        "NSE closes higher on positive momentum.",
        "FII flows remain supportive for key large caps.",
        "IT sector leads gains ahead of earnings season.",
    ]

    dashboard = [
        "----------------------------------------------------------",
        " TradeMind AI",
        "----------------------------------------------------------",
        "",
        f"Market Health          {'🟢' if health_score >= 75 else '🟡' if health_score >= 50 else '🔴'} {health_score}/100",
        "",
        f"NIFTY Trend            {nifty_trend}",
        "",
        f"BANKNIFTY              {bank_trend}",
        "",
        f"India VIX              {_vix_label(vix_level)}",
        "",
        f"Sector Leader          {sector_leader}",
        "",
        f"FII Activity           {fii_activity}",
        "",
        "Today's AI Picks",
        "",
    ]

    dashboard += [f"{symbol}" for symbol, _ in picks]
    dashboard += [
        "",
        "Open Positions",
        "",
        "Portfolio",
        "",
        "Performance",
        "",
        "Alerts",
        "",
        "News",
        "",
    ]
    dashboard += [f"- {headline}" for headline in headlines]
    return "\n".join(dashboard)


def summarize_performance(data: pd.DataFrame) -> str:
    if data.empty:
        return "No data available for analysis."

    if "Close" not in data.columns:
        return "No closing prices available for analysis."

    closing = data["Close"].dropna()
    if closing.empty:
        return "No closing prices available for analysis."

    start_price = float(closing.iloc[0])
    end_price = float(closing.iloc[-1])
    total_return = (end_price / start_price - 1) * 100
    returns = closing.pct_change().dropna()
    mean_daily = returns.mean() * 100
    volatility = returns.std() * 100
    latest_date = closing.index[-1].strftime("%Y-%m-%d")

    summary = [
        f"Stock analysis summary as of {latest_date}:",
        f"Start price: ${start_price:,.2f}",
        f"End price:   ${end_price:,.2f}",
        f"Total return: {total_return:+.2f}%",
        f"Average daily return: {mean_daily:+.4f}%",
        f"Daily volatility: {volatility:.4f}%",
    ]

    if "SMA_20" in data.columns and not pd.isna(data["SMA_20"].iloc[-1]):
        summary.append(f"20-day SMA: ${float(data['SMA_20'].iloc[-1]):,.2f}")
    if "SMA_50" in data.columns and not pd.isna(data["SMA_50"].iloc[-1]):
        summary.append(f"50-day SMA: ${float(data['SMA_50'].iloc[-1]):,.2f}")
    if "RSI_14" in data.columns and not pd.isna(data["RSI_14"].iloc[-1]):
        summary.append(f"RSI(14): {float(data['RSI_14'].iloc[-1]):.2f}")
    if "ATR_14" in data.columns and not pd.isna(data["ATR_14"].iloc[-1]):
        summary.append(f"ATR(14): {float(data['ATR_14'].iloc[-1]):.2f}")

    signal = identify_swing_trade_signal(data)
    if signal["signal"] != "hold":
        summary.extend([
            "",  # blank line before trade signal
            f"Trade signal: {signal['signal'].upper()}",
            f"Signal reason: {signal['reason']}",
        ])
        if signal["stop_loss"] is not None:
            summary.append(f"Suggested stop loss: ${signal['stop_loss']:.2f}")
        if signal["target"] is not None:
            summary.append(f"Suggested target: ${signal['target']:.2f}")
    else:
        summary.append("")
        summary.append("Trade signal: HOLD")
        summary.append(f"Signal reason: {signal['reason']}")

    return "\n".join(summary)

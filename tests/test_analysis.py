import pandas as pd
import pytest

from trademindai.analysis import (
    _normalize_symbol,
    add_technical_indicators,
    summarize_performance,
    generate_market_dashboard,
)


def test_summarize_performance_empty() -> None:
    empty_df = pd.DataFrame()
    result = summarize_performance(empty_df)
    assert result == "No data available for analysis."


def test_summarize_performance_closing_only() -> None:
    data = pd.DataFrame(
        {
            "Close": [100.0, 102.0, 101.0, 104.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
    )

    result = summarize_performance(data)
    assert "Start price: $100.00" in result
    assert "End price:   $104.00" in result
    assert "Total return: +4.00%" in result


def test_summarize_performance_no_close() -> None:
    data = pd.DataFrame(
        {
            "Open": [100.0, 102.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )

    result = summarize_performance(data)
    assert result == "No closing prices available for analysis."


def test_add_technical_indicators_generates_columns() -> None:
    data = pd.DataFrame(
        {
            "Close": [100.0] * 60,
        },
        index=pd.date_range("2026-01-01", periods=60, freq="D"),
    )

    result = add_technical_indicators(data)
    assert "SMA_20" in result.columns
    assert "SMA_50" in result.columns
    assert "RSI_14" in result.columns
    assert pd.isna(result["SMA_20"].iloc[18])
    assert not pd.isna(result["SMA_20"].iloc[19])
    assert pd.isna(result["SMA_50"].iloc[48])
    assert not pd.isna(result["SMA_50"].iloc[49])


def test_summarize_performance_includes_indicators() -> None:
    data = pd.DataFrame(
        {
            "Close": list(range(1, 61)),
        },
        index=pd.date_range("2026-01-01", periods=60, freq="D"),
    )
    data = add_technical_indicators(data)

    result = summarize_performance(data)
    assert "20-day SMA:" in result
    assert "50-day SMA:" in result
    assert "RSI(14):" in result


def test_normalize_nse_symbol() -> None:
    assert _normalize_symbol("TCS", "NSE") == "TCS.NS"
    assert _normalize_symbol("RELIANCE.NS", "NSE") == "RELIANCE.NS"
    assert _normalize_symbol("infy", "NSE") == "INFY.NS"
    assert _normalize_symbol("^NSEI", "NSE") == "^NSEI"


def test_generate_market_dashboard(monkeypatch) -> None:
    def fake_fetch_stock_data(symbol: str, period: str = "1mo", interval: str = "1d", exchange: str = "US"):
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        return pd.DataFrame({"Close": [100.0 + i for i in range(30)], "High": [100.0 + i for i in range(30)], "Low": [99.0 + i for i in range(30)]}, index=dates)

    monkeypatch.setattr('trademindai.analysis.fetch_stock_data', fake_fetch_stock_data)
    dashboard = generate_market_dashboard(exchange="NSE")
    assert "TradeMind AI" in dashboard
    assert "Market Health" in dashboard
    assert "Today's AI Picks" in dashboard

import argparse
import logging

from trademindai.analysis import (
    fetch_stock_data,
    summarize_performance,
    add_technical_indicators,
    generate_market_dashboard,
    scan_top_swing_picks,
    generate_trade_recommendation,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trademindai",
        description="TradeMind AI stock analysis toolkit",
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Stock ticker symbol to analyze (optional when using --dashboard)",
    )
    parser.add_argument(
        "--period",
        default="1mo",
        choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        help="History period to fetch",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        help="Data interval",
    )
    parser.add_argument(
        "--exchange",
        default="US",
        choices=["US", "NSE"],
        help="Exchange for the symbol (US or NSE)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show a market health dashboard instead of analyzing a single symbol",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Run the scanner and return top swing picks",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Generate a trade recommendation for the provided symbol",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.dashboard:
        logger.info("Generating market dashboard for %s", args.exchange)
        summary = generate_market_dashboard(exchange=args.exchange)
        print(summary)
        return 0

    if args.scan:
        logger.info("Scanning top swing picks for %s", args.exchange)
        picks = scan_top_swing_picks(exchange=args.exchange)
        for rank, pick in enumerate(picks, start=1):
            print(f"{rank}. {pick['symbol']} - Score: {pick['score']} - Signal: {pick['signal']}")
        return 0

    if args.recommend:
        if not args.symbol:
            raise SystemExit("Error: symbol is required for recommendation mode")
        logger.info("Generating trade recommendation for %s on %s", args.symbol, args.exchange)
        recommendation = generate_trade_recommendation(
            args.symbol,
            period=args.period,
            interval=args.interval,
            exchange=args.exchange,
        )
        for key, value in recommendation.items():
            print(f"{key}: {value}")
        return 0

    if not args.symbol:
        raise SystemExit("Error: symbol is required unless --dashboard, --scan, or --recommend is used")

    logger.info("Fetching stock data for %s on %s", args.symbol, args.exchange)
    data = fetch_stock_data(
        args.symbol,
        period=args.period,
        interval=args.interval,
        exchange=args.exchange,
    )
    data = add_technical_indicators(data)
    summary = summarize_performance(data)

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

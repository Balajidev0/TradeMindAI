# TradeMindAI

TradeMind AI is a stock analysis tool designed to help traders and investors evaluate equity performance with Python-driven analytics.

## Getting Started

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -U pip
python -m pip install -e .
```

3. Run the CLI:

```bash
trademindai --help
```

### NSE Usage

To analyze an Indian stock on NSE, pass `--exchange NSE`:

```bash
trademindai TCS --exchange NSE --period 1mo --interval 1d
```

The CLI will normalize the symbol to NSE format automatically.

### Dashboard Mode

Run a market health dashboard for NSE:

```bash
trademindai --dashboard --exchange NSE
```

### Scan Top Swing Picks

Run the scanner and return the top swing trade ideas:

```bash
trademindai --scan --exchange NSE
```

### Trade Recommendation

Generate a trade recommendation for a symbol:

```bash
trademindai --recommend BEL --exchange NSE --period 1mo --interval 1d
```

## Project Layout

- `src/trademindai/` — package source code
- `tests/` — unit tests
- `pyproject.toml` — project metadata and dependencies

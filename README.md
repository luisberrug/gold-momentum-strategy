# Gold Momentum Strategy

Research on a monthly gold allocation rule that combines lagged gold momentum with a cross asset signal derived from bonds. When both signals are positive, the strategy holds gold; otherwise it earns the cash rate.

The repository is structured to distinguish exploratory analysis from reusable, testable research code. It includes a walk forward procedure that selects lookback parameters using only a trailing training window and applies them to the next out of sample block.

## Research questions

- Does lagged gold momentum improve on a passive gold allocation on a risk adjusted basis?
- Does Treasury momentum provide a useful cross-asset filter?
- Do results remain directionally useful when lookbacks are selected through walk forward testing instead of full-sample optimization?

## Methodology

1. Build a monthly gold return history by combining spot gold history with GLD returns after the ETF becomes available.
2. Approximate 7-10 year Treasury returns from the average 7-year and 10-year yield, prior-month carry, and a duration-based price effect.
3. Validate the gold and Treasury proxies against GLD and IEF over overlapping periods.
4. Form lagged gold and Treasury momentum signals. Signals are shifted by one month to avoid using current-month information.
5. Compare buy and hold gold, a fixed parameter joint strategy, and a walk forward version.
6. Report CAGR, annualized volatility, Sharpe ratio, maximum drawdown, Calmar ratio, total return, and trade counts.

## Repository layout

```text
.
|-- notebooks/
|   `-- Gold_Momentum_Strategy.ipynb
|-- src/gold_momentum/
|   |-- __init__.py
|   `-- core.py
|-- tests/
|   `-- test_core.py
|-- DATA_AND_RISK_NOTICE.md
|-- pyproject.toml
`-- README.md
```

## Reproduce the analysis

Create an isolated environment, install the project, and run the tests:

```bash
python -m venv .venv
python -m pip install -e ".[notebook]"
python -m unittest discover -s tests -v
jupyter lab
```

Open `notebooks/Gold_Momentum_Strategy.ipynb` and run all cells from the repository root. The notebook downloads data at runtime; no third-party market data is stored in Git.

## Important limitations

- The synthetic Treasury series is a duration approximation, not a fully replicated bond index.
- The study does not model fees, spreads, market impact, taxes, or slippage.
- Public data providers can revise observations or change interfaces, so future runs may differ.
- The notebook intentionally contains no precomputed performance claims. Results should be regenerated in a clean environment immediately before publication.
- Backtests are research evidence, not forecasts or investment recommendations.

See [DATA_AND_RISK_NOTICE.md](DATA_AND_RISK_NOTICE.md) for data attribution and risk details.

## License

No software license has been selected in this draft. Add one only after the repository owner chooses the intended reuse terms.

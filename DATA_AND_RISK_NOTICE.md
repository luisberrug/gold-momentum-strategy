# Data and risk notice

This repository is an educational research project. It is not investment advice, a recommendation, or a production trading system. Historical and simulated performance does not guarantee future results.

The analysis downloads public market series at runtime:

- Monthly gold prices from the DataHub gold-prices data package.
- GLD and IEF adjusted market data through `yfinance`.
- DGS7, DGS10, and FEDFUNDS series from the Federal Reserve Economic Data service through `pandas-datareader`.

Users are responsible for reviewing the current licenses, attribution requirements, and usage terms of each upstream provider. No third-party data files are committed to this repository.

The synthetic Treasury return series is a first-order approximation based on carry and duration. It omits convexity, roll-down, fees, taxes, market impact, bid-ask spreads, and other implementation effects. Backtests are gross of transaction costs.

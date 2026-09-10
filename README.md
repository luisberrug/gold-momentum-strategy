# Gold Momentum Strategy

Research on a monthly gold allocation rule that combines lagged gold momentum with a cross asset signal derived from Treasury bonds. 

When both signals are positive, the strategy enters or maintains a long position in gold; otherwise it allocates to cash and earns the modeled monthly cash rate. Bond momentum acts as a filter, the portfolio does not hold the underlying.

The repository is structured to distinguish exploratory analysis from reusable, testable research code. It includes a walk-forward procedure that selects lookback parameters using only a trailing training window and applies them to the next out of sample block.

## Research questions

- Does lagged gold momentum improve on a passive gold allocation on a risk adjusted basis?
- Does Treasury bond momentum provide a useful signal for timing gold exposure?
- Does combining both signals improve the return and drawdown trade off?
- Does historical outperformance persist when lookbacks are selected through walk-forward testing?
- Do selected lookbacks become more stable over time?
- Are buy signals concentrated in particular calendar months or quarters?

## Methodology

1. Build a monthly gold return history by combining spot gold history with GLD returns after the ETF becomes available.
2. Approximate 7-10 year Treasury returns from the average 7 year and 10 year yield, prior month carry, and a duration based price effect.
3. Validate the gold and Treasury proxies against GLD and IEF over overlapping periods.
4. Form lagged gold and Treasury momentum signals. Signals are shifted by one month to avoid using current month information.
5. Compare the walk-forward strategy with buy & hold gold and a fixed parameter joint strategy.
6. Report CAGR, annualized volatility, Sharpe ratio, maximum drawdown, Calmar ratio, total return, and trade counts.

## Walk-forward design

At each iteration, the procedure:
1. Uses the preceding 20 years as the training sample.
2. Evaluates gold and Treasury lookbacks independently from 1, 3, 6, 9, and 12 months.
3. Selects the pair with the highest training period Sharpe ratio.
4. Applies that pair during the next 12 months.
5. Advances the window and repeats.
This avoids selecting each test window’s parameters using its subsequent returns. It provides a more realistic assessment of parameter selection than choosing the best fixed lookback after examining the full history.
Walk-forward testing does not eliminate all research bias: the signal design, candidate set, training length, and evaluation choices can still be influenced by prior examination of the data.

## Conclusions

The findings below summarize the research results reported for the evaluated sample. They should be confirmed against a fresh notebook run before publication, with the sample dates and corresponding performance table retained.
1. Historical outperformance with smaller drawdowns: The joint momentum approach outperformed B&H gold in the reported results while reducing maximum drawdown to approximately half the magnitude of the benchmark’s drawdown. This suggests that the bond filter may help improve the balance between participation in gold appreciation and exposure to prolonged losses. 
2. Walk-forward testing strengthens the research design: The full history comparison identifies which fixed lookbacks performed best retrospectively. Choosing a strategy solely from that comparison risks benefiting from hindsight. The walk-forward procedure addresses this parameter selection issue by using only the trailing training window to select each subsequent test period’s lookbacks. The out of sample comparison is therefore the central assessment of whether the approach remains useful beyond its training data.
3. Recent selections favor 6 month gold and 12 month Treasury momentum: In the latest reported walk-forward windows, the model consistently selected a 6 month gold lookback and a 12 month IEF lookback, this repeated selection suggests greater parameter stability in the recent sample and identifies Joint Momentum 6M/12M as a candidate for further evaluation. However, consecutive training windows overlap substantially, so repeated selections are not independent confirmations of an optimal specification. The pattern is consistent with possible stabilization around that parameter pair, but does not establish convergence to a permanently superior strategy.
4. Entry seasonality remains exploratory: The buy count analysis examines whether entries cluster by calendar month or quarter, with frequencies adjusted for the number of observed months. Entry clustering alone does not demonstrate an exploitable seasonal effect. Any seasonal rule would require a separate assessment of subsequent returns, trading costs, and out of sample performance before being incorporated into the strategy.

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

"""Core data preparation and backtesting functions.

The module separates network access from pure transformations so that the
strategy logic can be tested with synthetic data. It is research code, not an
execution or investment-advice system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd


GOLD_PRICE_URL = "https://datahub.io/core/gold-prices/r/monthly.csv"
TRADING_MONTHS = 12


@dataclass(frozen=True)
class SourceData:
    """Raw market series downloaded by :func:`download_source_data`."""

    gold_spot: pd.Series
    gld_daily: pd.Series
    ief_daily: pd.Series
    treasury_yields: pd.DataFrame
    fed_funds: pd.Series


def _month_start_index(index: pd.Index) -> pd.DatetimeIndex:
    values = pd.to_datetime(index)
    if getattr(values, "tz", None) is not None:
        values = values.tz_localize(None)
    return values.to_period("M").to_timestamp()


def _ticker_column(frame: pd.DataFrame, ticker: str, field: str = "Close") -> pd.Series:
    """Return one field across yfinance's single- or multi-index layouts."""

    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(0):
            selected = frame[ticker]
        elif ticker in frame.columns.get_level_values(-1):
            selected = frame.xs(ticker, axis=1, level=-1)
        else:
            raise KeyError(f"Ticker {ticker!r} was not present in downloaded columns")
    else:
        selected = frame
    if field not in selected.columns:
        raise KeyError(f"Field {field!r} was not present for ticker {ticker!r}")
    return selected[field].rename(ticker)


def download_source_data(
    start: str = "1969-12-31",
    end: date | str | None = None,
) -> SourceData:
    """Download the public market series used by the research notebook."""

    import pandas_datareader.data as web
    import yfinance as yf

    end_value = end or date.today()
    gld_frame = yf.download(
        "GLD",
        start="2000-01-01",
        end=end_value,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    ief_frame = yf.download(
        "IEF",
        start="2000-01-01",
        end=end_value,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    gold = pd.read_csv(
        GOLD_PRICE_URL,
        parse_dates=["Date"],
        index_col="Date",
    )["Price"].rename("Gold_Spot")
    yields = web.DataReader(["DGS7", "DGS10"], "fred", start, end_value)
    fed_funds = web.DataReader("FEDFUNDS", "fred", start, end_value)["FEDFUNDS"]
    return SourceData(
        gold_spot=gold,
        gld_daily=_ticker_column(gld_frame, "GLD"),
        ief_daily=_ticker_column(ief_frame, "IEF"),
        treasury_yields=yields,
        fed_funds=fed_funds,
    )


def splice_gold_returns(
    gold_spot: pd.Series,
    gld_daily: pd.Series,
    start: str = "1970-01-01",
) -> pd.Series:
    """Use spot-gold history, then switch to GLD when its return is available."""

    spot = gold_spot.astype(float).copy()
    spot.index = _month_start_index(spot.index)
    spot = spot.groupby(level=0).last().sort_index()

    gld = gld_daily.astype(float).copy().sort_index().resample("ME").last()
    gld.index = _month_start_index(gld.index)

    spot_return = spot.pct_change(fill_method=None)
    gld_return = gld.pct_change(fill_method=None)
    first_gld_return = gld_return.first_valid_index()
    if first_gld_return is None:
        raise ValueError("GLD did not contain enough observations to calculate returns")

    historical = spot_return.loc[(spot_return.index >= start) & (spot_return.index <= first_gld_return)]
    modern = gld_return.loc[gld_return.index > first_gld_return]
    combined = pd.concat([historical, modern]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    return combined.rename("Gold_Return")


def synthetic_bond_returns(
    treasury_yields: pd.DataFrame,
    duration: float = 7.5,
) -> pd.Series:
    """Approximate 7-10 year Treasury total returns from yield changes.

    The approximation combines prior-month carry with the first-order duration
    effect. It intentionally omits convexity, roll-down, fees, and trading costs.
    """

    required = {"DGS7", "DGS10"}
    missing = required.difference(treasury_yields.columns)
    if missing:
        raise KeyError(f"Missing Treasury yield columns: {sorted(missing)}")

    monthly = treasury_yields.loc[:, ["DGS7", "DGS10"]].astype(float)
    monthly = monthly.sort_index().resample("ME").last()
    monthly.index = _month_start_index(monthly.index)
    average_yield = monthly.mean(axis=1) / 100
    carry = average_yield.shift(1) / TRADING_MONTHS
    duration_effect = -duration * average_yield.diff()
    return (carry + duration_effect).rename("IEF_Return")


def build_research_dataset(source: SourceData, duration: float = 7.5) -> pd.DataFrame:
    """Create aligned monthly gold, synthetic Treasury, and cash returns."""

    gold_return = splice_gold_returns(source.gold_spot, source.gld_daily)
    bond_return = synthetic_bond_returns(source.treasury_yields, duration=duration)
    cash = source.fed_funds.astype(float).copy()
    cash.index = _month_start_index(cash.index)
    cash = cash.groupby(level=0).last().sort_index().div(100).rename("Cash_Rate")
    return pd.concat([gold_return, bond_return, cash], axis=1).sort_index()


def validate_proxies(source: SourceData, duration: float = 7.5) -> dict[str, float]:
    """Calculate return correlations for the GLD and synthetic-IEF proxies."""

    spot = source.gold_spot.astype(float).copy()
    spot.index = _month_start_index(spot.index)
    spot_return = spot.groupby(level=0).last().pct_change(fill_method=None)

    gld = source.gld_daily.astype(float).sort_index().resample("ME").last()
    gld.index = _month_start_index(gld.index)
    gld_return = gld.pct_change(fill_method=None)

    synthetic = synthetic_bond_returns(source.treasury_yields, duration=duration)
    ief = source.ief_daily.astype(float).sort_index().resample("ME").last()
    ief.index = _month_start_index(ief.index)
    ief_return = ief.pct_change(fill_method=None)

    return {
        "gold_spot_vs_gld": float(pd.concat([spot_return, gld_return], axis=1).dropna().corr().iloc[0, 1]),
        "synthetic_bond_vs_ief": float(pd.concat([synthetic, ief_return], axis=1).dropna().corr().iloc[0, 1]),
    }


def build_joint_strategy(
    data: pd.DataFrame,
    gold_lookback: int,
    bond_lookback: int,
) -> pd.DataFrame:
    """Invest in gold when both lagged momentum signals are positive; otherwise hold cash."""

    if gold_lookback < 1 or bond_lookback < 1:
        raise ValueError("Lookbacks must be positive integers")
    required = {"Gold_Return", "IEF_Return", "Cash_Rate"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing strategy columns: {sorted(missing)}")

    frame = data.loc[:, sorted(required)].astype(float).sort_index().copy()
    frame["Gold_Price"] = 100 * (1 + frame["Gold_Return"].fillna(0)).cumprod()
    frame["IEF_Price"] = 100 * (1 + frame["IEF_Return"].fillna(0)).cumprod()
    frame["Gold_Momentum"] = frame["Gold_Price"].pct_change(
        gold_lookback, fill_method=None
    ).shift(1)
    frame["IEF_Momentum"] = frame["IEF_Price"].pct_change(
        bond_lookback, fill_method=None
    ).shift(1)

    valid = frame[["Gold_Return", "Cash_Rate", "Gold_Momentum", "IEF_Momentum"]].notna().all(axis=1)
    frame = frame.loc[valid].copy()
    frame["Signal"] = (
        (frame["Gold_Momentum"] > 0) & (frame["IEF_Momentum"] > 0)
    ).astype(int)
    frame["Strategy_Return"] = (
        frame["Signal"] * frame["Gold_Return"]
        + (1 - frame["Signal"]) * (frame["Cash_Rate"] / TRADING_MONTHS)
    )
    frame["Equity"] = (1 + frame["Strategy_Return"]).cumprod()
    frame["Trade"] = frame["Signal"].diff().fillna(frame["Signal"])
    return frame


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series | None = None,
    signal: pd.Series | None = None,
    risk_free_returns: pd.Series | None = None,
) -> dict[str, float | int]:
    """Calculate annualized performance and trade-count statistics."""

    clean_returns = returns.dropna().astype(float)
    if clean_returns.empty:
        raise ValueError("At least one non-null return is required")
    if equity is None:
        clean_equity = (1 + clean_returns).cumprod()
    else:
        clean_equity = equity.reindex(clean_returns.index).dropna().astype(float)
        clean_returns = clean_returns.reindex(clean_equity.index)
    if len(clean_equity) < 2:
        raise ValueError("At least two equity observations are required")

    years = max((clean_equity.index[-1] - clean_equity.index[0]).days / 365.25, 1 / TRADING_MONTHS)
    total_return = clean_equity.iloc[-1] / clean_equity.iloc[0] - 1
    cagr = (clean_equity.iloc[-1] / clean_equity.iloc[0]) ** (1 / years) - 1
    excess = clean_returns
    if risk_free_returns is not None:
        excess = clean_returns - risk_free_returns.reindex(clean_returns.index).fillna(0)
    volatility = clean_returns.std() * np.sqrt(TRADING_MONTHS)
    sharpe = np.nan if excess.std() == 0 else excess.mean() / excess.std() * np.sqrt(TRADING_MONTHS)
    drawdown = clean_equity / clean_equity.cummax() - 1
    max_drawdown = drawdown.min()
    calmar = np.nan if max_drawdown == 0 else cagr / abs(max_drawdown)

    buys = sells = 0
    if signal is not None:
        trades = signal.reindex(clean_returns.index).fillna(0).astype(int).diff()
        buys = int((trades == 1).sum())
        sells = int((trades == -1).sum())

    return {
        "cagr": float(cagr),
        "annualized_volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "total_return": float(total_return),
        "buys": buys,
        "sells": sells,
    }


def select_best_lookbacks(
    training_data: pd.DataFrame,
    candidates: Iterable[int],
) -> tuple[tuple[int, int], float]:
    """Select the gold/IEF lookback pair with the highest in-sample Sharpe."""

    candidate_list = sorted(set(int(value) for value in candidates))
    if not candidate_list or candidate_list[0] < 1:
        raise ValueError("Candidates must contain positive integers")

    best_pair = (candidate_list[0], candidate_list[0])
    best_score = -np.inf
    for gold_lookback in candidate_list:
        for bond_lookback in candidate_list:
            strategy = build_joint_strategy(training_data, gold_lookback, bond_lookback)
            returns = strategy["Strategy_Return"]
            score = -np.inf if returns.std() == 0 else returns.mean() / returns.std() * np.sqrt(TRADING_MONTHS)
            if score > best_score:
                best_pair = (gold_lookback, bond_lookback)
                best_score = float(score)
    return best_pair, best_score


def walk_forward_backtest(
    data: pd.DataFrame,
    candidates: Iterable[int] = (1, 3, 6, 9, 12),
    train_months: int = 240,
    test_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune on a trailing window and apply parameters to the next test block."""

    if train_months < 2 or test_months < 1:
        raise ValueError("train_months must be at least 2 and test_months must be positive")
    ordered = data.sort_index().copy()
    if len(ordered) <= train_months:
        raise ValueError("The data set must be longer than the training window")

    output = pd.DataFrame(index=ordered.index)
    records: list[dict[str, object]] = []
    start = train_months
    while start < len(ordered):
        end = min(start + test_months, len(ordered))
        training = ordered.iloc[start - train_months : start]
        pair, training_sharpe = select_best_lookbacks(training, candidates)
        gold_lookback, bond_lookback = pair

        history_through_test = ordered.iloc[:end]
        candidate = build_joint_strategy(history_through_test, gold_lookback, bond_lookback)
        test_index = ordered.iloc[start:end].index
        available = test_index.intersection(candidate.index)
        output.loc[available, "Strategy_Return"] = candidate.loc[available, "Strategy_Return"]
        output.loc[available, "Signal"] = candidate.loc[available, "Signal"]
        output.loc[available, "Gold_Lookback"] = gold_lookback
        output.loc[available, "IEF_Lookback"] = bond_lookback
        records.append(
            {
                "test_start": test_index[0],
                "test_end": test_index[-1],
                "gold_lookback": gold_lookback,
                "ief_lookback": bond_lookback,
                "training_sharpe": training_sharpe,
            }
        )
        start = end

    output = output.dropna(subset=["Strategy_Return"]).copy()
    output["Signal"] = output["Signal"].astype(int)
    output["Equity"] = (1 + output["Strategy_Return"]).cumprod()
    output["Trade"] = output["Signal"].diff().fillna(output["Signal"])
    return output, pd.DataFrame.from_records(records)

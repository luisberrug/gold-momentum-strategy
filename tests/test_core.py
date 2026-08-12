import unittest

import numpy as np
import pandas as pd

from gold_momentum import build_joint_strategy, compute_metrics, walk_forward_backtest


def sample_data(periods: int = 48) -> pd.DataFrame:
    index = pd.date_range("2000-01-01", periods=periods, freq="MS")
    repetitions = int(np.ceil(periods / 4))
    gold = pd.Series(np.tile([0.02, 0.01, -0.01, 0.03], repetitions)[:periods], index=index)
    bond = pd.Series(np.tile([0.01, 0.005, -0.002, 0.008], repetitions)[:periods], index=index)
    return pd.DataFrame(
        {
            "Gold_Return": gold,
            "IEF_Return": bond,
            "Cash_Rate": 0.03,
        }
    )


class CoreTests(unittest.TestCase):
    def test_signal_uses_lagged_momentum(self) -> None:
        data = sample_data()
        strategy = build_joint_strategy(data, gold_lookback=1, bond_lookback=1)
        first_date = strategy.index[0]
        prior_date = data.index[data.index.get_loc(first_date) - 1]
        expected_gold_momentum = data.loc[prior_date, "Gold_Return"]
        self.assertTrue(np.isclose(strategy.loc[first_date, "Gold_Momentum"], expected_gold_momentum))

    def test_metrics_are_finite_for_nonconstant_returns(self) -> None:
        data = sample_data()
        strategy = build_joint_strategy(data, 3, 3)
        metrics = compute_metrics(
            strategy["Strategy_Return"],
            strategy["Equity"],
            strategy["Signal"],
        )
        self.assertTrue(np.isfinite(metrics["cagr"]))
        self.assertTrue(np.isfinite(metrics["sharpe"]))
        self.assertLessEqual(metrics["max_drawdown"], 0)

    def test_future_changes_do_not_alter_first_walk_forward_block(self) -> None:
        data = sample_data()
        baseline, _ = walk_forward_backtest(
            data,
            candidates=(1, 3),
            train_months=18,
            test_months=6,
        )
        changed = data.copy()
        changed.loc[changed.index[-6:], "Gold_Return"] = -0.25
        revised, _ = walk_forward_backtest(
            changed,
            candidates=(1, 3),
            train_months=18,
            test_months=6,
        )
        pd.testing.assert_series_equal(
            baseline["Strategy_Return"].iloc[:6],
            revised["Strategy_Return"].iloc[:6],
        )


if __name__ == "__main__":
    unittest.main()

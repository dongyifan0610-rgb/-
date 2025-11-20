"""Factor definitions for synthetic backtests without external deps.

This module intentionally keeps the implementation extremely explicit and
verbose so the learning process behind each factor is transparent. The comments
walk through every transformation step-by-step, which makes it easier to extend
or tweak individual building blocks when experimenting with new ideas.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Dict, Iterable, List, Sequence


@dataclass
class PricePanel:
    prices: List[List[float]]  # rows: days, cols: assets
    assets: List[str]

    def last(self) -> List[float]:
        """Return the most recent price vector (latest day).

        Keeping this helper explicit avoids repetitive ``panel.prices[-1]``
        calls in factor implementations and makes the intent clear.
        """
        return self.prices[-1]

    def slice_window(self, window: int) -> List[List[float]]:
        """Return the trailing ``window`` days of prices.

        The backtest uses fixed-length lookbacks for momentum, mean reversion,
        and volatility. Providing a dedicated helper keeps those calculations
        symmetrical and easy to audit.
        """
        return self.prices[-window:]

    @property
    def num_assets(self) -> int:
        return len(self.assets)


@dataclass
class Factor:
    name: str
    description: str
    fn: Callable[[PricePanel], Dict[str, float]]

    def compute(self, panel: PricePanel) -> Dict[str, float]:
        """Execute the factor function on the provided price panel.

        A thin wrapper is used instead of exposing ``fn`` directly so callers
        have a consistent interface and so we can add logging/instrumentation in
        one place if needed.
        """
        return self.fn(panel)


def _pct_change_last(panel: PricePanel, periods: int = 5) -> Dict[str, float]:
    """Helper to compute percentage change over ``periods``.

    A standalone helper keeps the momentum factor concise and also makes it
    reusable by other signals, such as the few-shot generator when it wants to
    incorporate price-change style operations.
    """

    # ``prev`` is the price vector ``periods`` days ago (e.g., five-day lookback
    # if ``periods=5``). ``last`` is the most recent price vector.
    prev = panel.prices[-periods]
    last = panel.last()

    # For each asset, compute (current / past) - 1 which yields a simple return.
    return {
        asset: (last[idx] / prev[idx]) - 1
        for idx, asset in enumerate(panel.assets)
    }


def momentum_factor(window: int = 20) -> Factor:
    def compute(panel: PricePanel) -> Dict[str, float]:
        return _pct_change_last(panel, periods=window)

    return Factor(
        name=f"momentum_{window}",
        description=f"{window}-day price momentum",
        fn=compute,
    )


def mean_reversion_factor(window: int = 10) -> Factor:
    def compute(panel: PricePanel) -> Dict[str, float]:
        # Pull the trailing window so every asset uses identical lookback length.
        window_prices = panel.slice_window(window)

        # Compute a per-asset mean across the selected window. The comprehension
        # walks column-by-column (asset-by-asset) rather than day-by-day so the
        # intent is easy to follow.
        means = [mean([row[i] for row in window_prices]) for i in range(panel.num_assets)]

        # Current prices, used to measure divergence from the mean.
        last = panel.last()

        # Positive values indicate prices are above the mean (potentially rich),
        # negative values indicate they are below (potentially cheap).
        return {
            asset: (last[idx] - means[idx]) / means[idx]
            for idx, asset in enumerate(panel.assets)
        }

    return Factor(
        name=f"mean_reversion_{window}",
        description=f"{window}-day mean reversion",
        fn=compute,
    )


def volatility_factor(window: int = 20) -> Factor:
    def compute(panel: PricePanel) -> Dict[str, float]:
        # We need ``window + 1`` rows to compute ``window`` returns. Each return
        # requires a current and previous price, so a 20-day volatility uses 21
        # daily prices.
        window_prices = panel.slice_window(window + 1)
        vols: List[float] = []

        # Compute standard deviation of simple daily returns for each asset.
        for asset_idx in range(panel.num_assets):
            returns: List[float] = []
            for day in range(1, len(window_prices)):
                prev = window_prices[day - 1][asset_idx]
                cur = window_prices[day][asset_idx]
                returns.append((cur / prev) - 1)

            # Population standard deviation (pstdev) matches the deterministic
            # nature of synthetic data. If we have fewer than two returns, fall
            # back to 0.0 to keep the signal defined.
            vols.append(pstdev(returns) if len(returns) > 1 else 0.0)

        return {asset: vols[idx] for idx, asset in enumerate(panel.assets)}

    return Factor(
        name=f"volatility_{window}",
        description=f"{window}-day volatility",
        fn=compute,
    )


EXAMPLE_FACTORS: Dict[str, Factor] = {
    "momentum": momentum_factor(),
    "mean_reversion": mean_reversion_factor(),
    "volatility": volatility_factor(),
}


def apply_factors(panel: PricePanel, factors: Iterable[Factor]) -> Dict[str, Dict[str, float]]:
    """Compute factor values for the latest date.

    Each factor is executed independently so their outputs remain comparable and
    easy to debug. The returned dictionary is nested by factor name, then asset
    ticker, to mirror common practitioner data layouts.
    """

    results: Dict[str, Dict[str, float]] = {}
    for factor in factors:
        results[factor.name] = factor.compute(panel)
    return results

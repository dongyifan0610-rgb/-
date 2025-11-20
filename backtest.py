"""Run a toy backtest with few-shot generated factors using only stdlib.

The script is deliberately verbose and annotated in detail. Each section is
broken down to show how prices are synthesized, signals are computed, positions
are derived, and returns are measured. This makes the end-to-end flow easy to
follow for readers new to factor backtesting.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List

from factors import EXAMPLE_FACTORS, Factor, PricePanel, apply_factors
from few_shot_generator import generate_new_factor


@dataclass
class BacktestResult:
    returns: Dict[str, float]
    cumulative: Dict[str, float]
    factor_returns: Dict[str, float]


def make_synthetic_prices(num_assets: int = 50, num_days: int = 252) -> PricePanel:
    """Create a deterministic price panel using geometric Brownian motion.

    * ``num_assets`` controls the cross-section size.
    * ``num_days`` controls the length of the price history.
    * A fixed RNG seed makes the example reproducible for teaching.
    """

    rng = Random(42)
    prices: List[List[float]] = []

    # Start every asset at the same level to highlight factor behavior rather
    # than idiosyncratic starting points.
    current = [100.0 for _ in range(num_assets)]

    for _ in range(num_days):
        daily = []
        for price in current:
            # Draw a small random shock; mean drift 5 bps with 1% daily vol.
            shock = rng.gauss(0.0005, 0.01)
            new_price = price * (1 + shock)
            daily.append(new_price)

        # Append the simulated day and roll forward the "current" vector.
        prices.append(daily)
        current = daily

    # Create simple asset identifiers for readability in printed results.
    assets = [f"Asset_{i:02d}" for i in range(num_assets)]
    return PricePanel(prices=prices, assets=assets)


def compute_positions(signals: Dict[str, Dict[str, float]], top_quantile: float = 0.2) -> Dict[str, Dict[str, float]]:
    """Translate factor scores into long/short weights.

    A simple quantile portfolio is used: go long the top ``top_quantile``
    fraction, short the bottom ``top_quantile`` fraction, and scale weights so
    the gross exposure sums to 1.0 (0.5 long / 0.5 short). This mirrors common
    cross-sectional testing conventions.
    """

    positions: Dict[str, Dict[str, float]] = {}
    for factor_name, values in signals.items():
        # Sort assets by factor value ascending so the first ``count`` are the
        # lowest scores and the last ``count`` are the highest.
        sorted_assets = sorted(values.items(), key=lambda kv: kv[1])
        count = max(1, int(len(sorted_assets) * top_quantile))
        longs = {asset for asset, _ in sorted_assets[-count:]}
        shorts = {asset for asset, _ in sorted_assets[:count]}

        # Assign equal weight inside each side of the book; neutral elsewhere.
        book: Dict[str, float] = {}
        for asset, _ in sorted_assets:
            if asset in longs:
                book[asset] = 1.0 / (2 * count)
            elif asset in shorts:
                book[asset] = -1.0 / (2 * count)
            else:
                book[asset] = 0.0
        positions[factor_name] = book
    return positions


def run_backtest(panel: PricePanel, factors: List[Factor]) -> BacktestResult:
    """Evaluate a single-period cross-sectional backtest.

    为了便于理解，这里将传统的多期回测压缩为一步：
    1. 先计算所有因子在最新时点的打分。
    2. 基于分位数规则构建多空权重。
    3. 使用最近两个交易日的价格计算当期收益。
    尽管简单，但涵盖了端到端的核心骨架，便于教学与调试。
    """

    factor_values = apply_factors(panel, factors)
    positions = compute_positions(factor_values)

    # Use last two days to create a single holding-period return
    last_prices = panel.prices[-1]
    prev_prices = panel.prices[-2]
    daily_returns = [last / prev - 1 for last, prev in zip(last_prices, prev_prices)]

    factor_returns: Dict[str, float] = {}
    cumulative: Dict[str, float] = {}
    for factor_name, weights in positions.items():
        ret = sum(weights[asset] * daily_returns[idx] for idx, asset in enumerate(panel.assets))
        factor_returns[factor_name] = ret
        cumulative[factor_name] = (1 + ret)

    return BacktestResult(returns=factor_returns, cumulative=cumulative, factor_returns=factor_returns)


def main():
    panel = make_synthetic_prices()
    new_factor = generate_new_factor()
    factor_list = list(EXAMPLE_FACTORS.values()) + [new_factor]

    result = run_backtest(panel, factor_list)

    print("=== Factor Level Returns (1-period demonstration) ===")
    for name, ret in result.factor_returns.items():
        print(f"{name:25s}: {ret: .4%}")

    print("\n=== Cumulative PnL for each factor ===")
    for name, value in result.cumulative.items():
        print(f"{name:25s}: {value: .4f}")


if __name__ == "__main__":
    main()

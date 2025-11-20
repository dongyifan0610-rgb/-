"""Few-shot inspired factor generator without third-party libraries.

The goal of this module is pedagogical: show how a tiny amount of metadata from
existing factors can be remixed into a new composite signal. Every step is
documented so the reasoning behind the few-shot style "imitation" is clear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from factors import Factor, PricePanel, EXAMPLE_FACTORS


def _normalize(values: List[float]) -> List[float]:
    """Zero-mean, unit-variance scaling for a vector.

    Normalization ensures the generated signals share the same magnitude before
    any optional volatility scaling is applied. The implementation is kept
    manual (instead of relying on numpy/pandas) to keep the dependency surface
    minimal while still making the math obvious.
    """
    if not values:
        return []
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / max(len(values), 1)
    std = var ** 0.5 if var > 0 else 1.0
    return [(v - mu) / std for v in values]


@dataclass
class FactorPrototype:
    name: str
    description: str
    operations: List[str]


def generate_prototype(examples: Sequence[Factor]) -> FactorPrototype:
    """Create a lightweight prototype from the example factor descriptions.

    The function mimics the essence of few-shot learning by "observing" the
    descriptive language attached to baseline factors and mapping those cues to
    a set of primitive operations. It purposely avoids statistical fitting to
    keep the logic transparent and easy to follow.
    """

    # 1) Extract keyword-style tokens from the human-readable descriptions.
    tokens = []
    for factor in examples:
        tokens.extend(factor.description.split())
    keywords = {token.strip(",").lower() for token in tokens if len(token) > 4}

    # 2) Translate keywords into building blocks for the generated factor.
    operations: List[str] = []
    if "momentum" in keywords:
        operations.append("price_change")
    if "volatility" in keywords or "variance" in keywords:
        operations.append("volatility_scale")
    if "mean" in keywords:
        operations.append("mean_reversion")

    # Fallback: always include a price-change view so the prototype is non-empty.
    if not operations:
        operations.append("price_change")

    # 3) Package the derived operations into a prototype description.
    return FactorPrototype(
        name="few_shot_composite",
        description="Composite factor generated from example metadata",
        operations=operations,
    )


def build_factor_from_prototype(proto: FactorPrototype) -> Factor:
    """Convert a prototype description into a concrete Factor instance.

    The resulting factor mixes one or more signal legs, normalizes them, and
    optionally volatility-scales the output. Each block is heavily commented to
    make the decision path from prototype to calculation explicit.
    """

    def compute(panel: PricePanel) -> dict:
        # Accumulate raw signals and their weights so we can blend them later.
        signals: List[List[float]] = []
        weights: List[float] = []
        last = panel.last()

        # Momentum-style leg: simple price change over 5 days (chosen for
        # illustration). Weight 0.5 to make it the dominant component.
        if "price_change" in proto.operations:
            ref = panel.prices[-5]
            signals.append([(last[i] / ref[i]) - 1 for i in range(panel.num_assets)])
            weights.append(0.5)

        # Mean-reversion leg: negate the distance from a 15-day mean to express
        # a preference for assets below their average. Weighted lower than
        # momentum so it acts as a stabilizer rather than the main driver.
        if "mean_reversion" in proto.operations:
            window = panel.slice_window(15)
            means = [sum(row[i] for row in window) / len(window) for i in range(panel.num_assets)]
            signals.append([-(last[i] - means[i]) / means[i] for i in range(panel.num_assets)])
            weights.append(0.3)

        # If no operations were mapped, return a neutral (zero) vector to keep
        # the API contract predictable.
        if not signals:
            return {asset: 0.0 for asset in panel.assets}

        # Blend weighted signals component-wise. Using an explicit double loop
        # keeps the math easy to audit without relying on vectorized operations.
        blended: List[float] = [0.0 for _ in range(panel.num_assets)]
        for signal, weight in zip(signals, weights):
            for i, value in enumerate(signal):
                blended[i] += weight * value

        # Normalize to unit scale so downstream scaling decisions are clear and
        # independent of the raw signal magnitudes.
        normalized = _normalize(blended)

        # Optional volatility scaling dampens signals on assets with higher
        # variance, mimicking risk-adjusted scoring.
        if "volatility_scale" in proto.operations:
            window = panel.slice_window(21)
            vols: List[float] = []
            for asset_idx in range(panel.num_assets):
                returns = []
                for day in range(1, len(window)):
                    prev = window[day - 1][asset_idx]
                    cur = window[day][asset_idx]
                    returns.append((cur / prev) - 1)
                var = sum(r ** 2 for r in returns) / max(len(returns), 1)
                vols.append(var ** 0.5 if var > 0 else 1.0)
            normalized = [val / vols[i] for i, val in enumerate(normalized)]

        return {asset: normalized[idx] for idx, asset in enumerate(panel.assets)}

    return Factor(name=proto.name, description=proto.description, fn=compute)


def generate_new_factor() -> Factor:
    """High-level helper that produces the few-shot-inspired factor.

    将示例因子当作“提示”，先提炼原型，再转换成可执行的因子对象。
    这样调用者无需关心细节，直接获得可用于回测的信号函数。
    """

    examples = list(EXAMPLE_FACTORS.values())
    proto = generate_prototype(examples)
    return build_factor_from_prototype(proto)

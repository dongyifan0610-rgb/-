# Few-Shot Factor Generation Demo

This repository demonstrates how to create a new factor via a lightweight
few-shot-inspired process and evaluate it in a toy backtest. Everything runs on
synthetic data and uses only the Python standard library so you can run it
anywhere.

1. Define baseline factors.
2. Generate a new factor by mimicking the style of the examples.
3. Run a simple long/short backtest.

The source files now include逐行注释 (line-by-line commentary) to explain how
each step works and why certain choices were made (lookback windows, weighting,
normalization, volatility scaling, etc.). They can be read as a narrative guide
to constructing and testing factors without external dependencies.

## Files
- `factors.py`: Example factor definitions (momentum, mean reversion, volatility)
  and utilities for applying them.
- `few_shot_generator.py`: Builds a new factor prototype from the example
  metadata and converts it into a concrete factor.
- `backtest.py`: End-to-end script for synthesizing prices, generating the new
  factor, and running a cross-sectional backtest.

## How to run
Execute the script with Python 3:

```bash
python backtest.py
```

The script prints each factor's return over a single holding period and the
cumulative PnL of an equal-weight book combining them.

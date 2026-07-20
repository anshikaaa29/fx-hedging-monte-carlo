"""
generate_fx_data.py
--------------------
Generates realistic, noisy mock historical EUR/USD daily FX data for use in the
FX-Adjusted Trade Finance & Hedging Model.

Method:
- Daily log returns are simulated using a GARCH(1,1) volatility process, which
  reproduces the "volatility clustering" seen in real FX markets (calm periods
  followed by turbulent periods), layered on top of a small drift term.
- Output: eurusd_historical.csv with columns [Date, EURUSD]

RUN THIS SCRIPT FIRST. It creates the CSV consumed by fx_hedging_monte_carlo.py.
"""

import numpy as np
import pandas as pd

# ----------------------------
# Reproducibility
# ----------------------------
np.random.seed(42)

# ----------------------------
# Parameters (calibrated to realistic EUR/USD behavior)
# ----------------------------
N_DAYS = 730                  # 2 years of daily data
S0 = 1.0850                   # Starting EUR/USD spot rate
MU_ANNUAL = 0.005             # 0.5% annual drift (FX is close to a random walk)
SIGMA_ANNUAL = 0.08           # 8% annualized volatility (typical for EUR/USD)

TRADING_DAYS = 252
mu_daily = MU_ANNUAL / TRADING_DAYS
sigma_daily_long_run = SIGMA_ANNUAL / np.sqrt(TRADING_DAYS)

# GARCH(1,1) parameters -> creates realistic volatility clustering
# variance_t = omega + alpha * shock_(t-1)^2 + beta * variance_(t-1)
alpha = 0.06     # weight on yesterday's shock (reaction)
beta = 0.90      # weight on yesterday's variance (persistence)
long_run_var = sigma_daily_long_run ** 2
omega = long_run_var * (1 - alpha - beta)

# ----------------------------
# Simulate returns with GARCH(1,1) stochastic volatility
# ----------------------------
returns = np.zeros(N_DAYS)
variances = np.zeros(N_DAYS)
variances[0] = long_run_var

z = np.random.standard_normal(N_DAYS)  # random shocks

for t in range(1, N_DAYS):
    variances[t] = omega + alpha * (returns[t - 1] ** 2) + beta * variances[t - 1]
    returns[t] = mu_daily + np.sqrt(variances[t]) * z[t]

# ----------------------------
# Build price path (cumulative log returns -> price levels)
# ----------------------------
log_prices = np.log(S0) + np.cumsum(returns)
prices = np.exp(log_prices)

# ----------------------------
# Assemble and save DataFrame
# ----------------------------
dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=N_DAYS + 10)[-N_DAYS:]
df = pd.DataFrame({"Date": dates, "EURUSD": np.round(prices, 5)})

df.to_csv("eurusd_historical.csv", index=False)

print(f"Generated {N_DAYS} days of mock EUR/USD data -> eurusd_historical.csv")
print(df.tail())

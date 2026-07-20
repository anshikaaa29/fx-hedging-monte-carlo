"""
fx_hedging_monte_carlo.py
--------------------------
FX-Adjusted Trade Finance & Hedging Model
Monte Carlo simulation of margin erosion on a EUR-denominated trade receivable,
comparing an unhedged position to one partially hedged with an FX forward.

RUN generate_fx_data.py FIRST to produce eurusd_historical.csv.
(If the file isn't found, this script will generate it automatically.)

Business scenario:
  A US-based exporter invoices a customer EUR 500,000, payable in 90 days.
  The deal was priced with an 8% profit margin baked in at quote-time spot.
  Because the USD cost base is fixed, EUR depreciation over the 90-day window
  erodes (or wipes out) that margin when the EUR receivable is converted to USD.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(7)

# =========================================================
# 0. Load (or generate) historical FX data
# =========================================================
DATA_FILE = "eurusd_historical.csv"

if not os.path.exists(DATA_FILE):
    print(f"'{DATA_FILE}' not found -- generating it now...")
    os.system("python3 generate_fx_data.py")

hist = pd.read_csv(DATA_FILE, parse_dates=["Date"])

# =========================================================
# 1. Estimate drift (mu) and volatility (sigma) from history
# =========================================================
hist["log_ret"] = np.log(hist["EURUSD"] / hist["EURUSD"].shift(1))
hist = hist.dropna()

TRADING_DAYS = 252
mu_daily = hist["log_ret"].mean()
sigma_daily = hist["log_ret"].std()

mu_annual = mu_daily * TRADING_DAYS
sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS)

spot0 = hist["EURUSD"].iloc[-1]

print("=== Calibration from historical data ===")
print(f"Spot (S0):           {spot0:.4f}")
print(f"Annualized drift:    {mu_annual:.4%}")
print(f"Annualized vol:      {sigma_annual:.4%}\n")

# =========================================================
# 2. Deal parameters
# =========================================================
INVOICE_EUR = 500_000        # Trade receivable, invoiced in EUR
MARGIN_PCT = 0.08            # Profit margin baked in at quote-time spot
HORIZON_DAYS = 90            # Payment terms
N_PATHS = 1_000               # Monte Carlo paths
HEDGE_RATIO = 0.15            # Fraction of exposure hedged via FX forward

# Fixed USD cost basis -- locked in regardless of how FX moves
cost_basis_usd = INVOICE_EUR * spot0 * (1 - MARGIN_PCT)
expected_profit_usd = INVOICE_EUR * spot0 * MARGIN_PCT

# Simple forward rate via covered interest rate parity: spot adjusted for a
# small EUR-USD interest rate differential (~1.5% annualized carry).
RATE_DIFFERENTIAL = 0.015
forward_rate = spot0 * (1 + RATE_DIFFERENTIAL * HORIZON_DAYS / 365)

# =========================================================
# 3. Monte Carlo: simulate 1,000 GBM paths, 90 days each
# =========================================================
n_steps = HORIZON_DAYS

# Random shocks: shape (N_PATHS, n_steps)
Z = np.random.standard_normal((N_PATHS, n_steps))

daily_drift = mu_daily - 0.5 * sigma_daily ** 2   # Ito correction for GBM
log_ret_paths = daily_drift + sigma_daily * Z
cum_log_ret = np.cumsum(log_ret_paths, axis=1)
price_paths = spot0 * np.exp(cum_log_ret)

# Terminal FX rate at day 90 for every path
S_T = price_paths[:, -1]

# =========================================================
# 4. Unhedged vs. hedged outcomes
# =========================================================
# Unhedged: 100% of the EUR receivable converts at the day-90 spot rate
revenue_unhedged = INVOICE_EUR * S_T
profit_unhedged = revenue_unhedged - cost_basis_usd

# Hedged: HEDGE_RATIO of the receivable is locked at the forward rate;
# the remainder still floats with the spot rate at settlement.
revenue_hedged = (HEDGE_RATIO * INVOICE_EUR * forward_rate) + \
                  ((1 - HEDGE_RATIO) * INVOICE_EUR * S_T)
profit_hedged = revenue_hedged - cost_basis_usd

# Margin erosion (%) relative to the profit expected at quote-time spot
erosion_unhedged = (expected_profit_usd - profit_unhedged) / expected_profit_usd
erosion_hedged = (expected_profit_usd - profit_hedged) / expected_profit_usd

# =========================================================
# 5. Risk statistics
# =========================================================
def summarize(name, profit, erosion):
    var_5 = np.percentile(profit, 5)               # 95% Value-at-Risk
    cvar_5 = profit[profit <= var_5].mean()          # Expected shortfall beyond VaR
    prob_margin_wiped = (profit <= 0).mean()          # P(deal becomes unprofitable)
    print(f"--- {name} ---")
    print(f"Mean profit (USD):          {profit.mean():,.0f}")
    print(f"Std dev of profit (USD):    {profit.std():,.0f}")
    print(f"95% VaR (USD):               {var_5:,.0f}")
    print(f"95% CVaR / Exp. Shortfall:   {cvar_5:,.0f}")
    print(f"Mean margin erosion:         {erosion.mean():.2%}")
    print(f"P(margin fully wiped out):   {prob_margin_wiped:.2%}\n")
    return profit.std()

print("=== Simulation Results (1,000 paths, 90-day horizon) ===\n")
std_unhedged = summarize("UNHEDGED", profit_unhedged, erosion_unhedged)
std_hedged = summarize(f"HEDGED ({HEDGE_RATIO:.0%} forward)", profit_hedged, erosion_hedged)

variance_reduction = 1 - (std_hedged / std_unhedged)
print("=== Hedge Effectiveness ===")
print(f"Reduction in USD margin volatility from hedge: {variance_reduction:.1%}")
print("(Locking a fixed share of the exposure at a forward rate removes that same")
print(" share of variance one-for-one -- hedging 15% of the receivable cuts outcome")
print(" volatility by ~15%, which is the effect quantified above.)\n")

# =========================================================
# 6. Visualization
# =========================================================
plt.figure(figsize=(10, 6))
plt.hist(profit_unhedged, bins=40, alpha=0.6, label="Unhedged", color="#d9534f")
plt.hist(profit_hedged, bins=40, alpha=0.6, label=f"Hedged ({HEDGE_RATIO:.0%})", color="#5cb85c")
plt.axvline(expected_profit_usd, color="black", linestyle="--", label="Expected margin at quote")
plt.axvline(0, color="gray", linestyle=":", label="Break-even")
plt.title("Monte Carlo Simulation: USD Margin Outcome on \u20ac500,000 Invoice (90-Day Horizon)")
plt.xlabel("Realized USD Profit")
plt.ylabel("Frequency (out of 1,000 paths)")
plt.legend()
plt.tight_layout()
plt.savefig("fx_margin_erosion_simulation.png", dpi=150)
print("Chart saved -> fx_margin_erosion_simulation.png")

###############################################################################
# EMPIRICAL METHODS IN FINANCE 2025
# =============================================================================
# GROUP MEMBERS:
# Daniel Vito Lobasso
# Thomas Nava
# Jacopo Sinigaglia
# Elvedin Muminovic
# =============================================================================
# Project #1: "Cointegration and Pair Trading"
# Goal:  Design a statistical arbitrage strategy by implementing pair trading.
###############################################################################

# The code is optimized for Python 3.11.

###############################################################################

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, norm, gaussian_kde
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey, het_white
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
# import scipy.stats as stats
from scipy.stats import t
from itertools import permutations

###############################################################################
# PLOT FUNCTION FOR TIME SERIES
###############################################################################
def plot_time_series(returns_df, title):
    plt.figure(figsize=(12,6))
    for col in returns_df.columns:
        plt.plot(returns_df.index, returns_df[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()

###############################################################################
# PART 0: DIRECTORY AND DATA SET UP
###############################################################################

# Set working directory to the script's location
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(base_dir)
sys.path.insert(0, base_dir)
print("Current working directory:", os.getcwd())

# File paths
input_file = os.path.join("Data", "Data_Project1.xlsx")
output_file = os.path.join("Data", "Entertainment_data.xlsx")

# Check if the output file already exists. If not, create it.
if not os.path.exists(output_file):
    # Read the entire sheet without a header
    df_entertainment = pd.read_excel(input_file, sheet_name="Entertainment", header=None)
    
    # Eliminate the first two rows and reset the index
    df_entertainment = df_entertainment.iloc[2:].reset_index(drop=True)
    
    # Assign new header columns: first column for dates, next five for company codes
    df_entertainment.columns = ["Date", "ATVI", "NNDO", "SONY", "EA", "MSFT"]
    
    # Convert the 'Date' column to datetime (YYYY-mm-dd)
    df_entertainment["Date"] = pd.to_datetime(df_entertainment["Date"], format="%Y-%m-%d")
    
    # Save the processed DataFrame to the output file
    df_entertainment.to_excel(output_file, sheet_name="Entertainment", index=False)
    print(f"'Entertainment' sheet has been saved to {output_file}")
else:
    print(f"Entertainment_data file already exists at {output_file}, skipping creation.")
    # Load the existing file
    df_entertainment = pd.read_excel(output_file, sheet_name="Entertainment")
    df_entertainment["Date"] = pd.to_datetime(df_entertainment["Date"], format="%Y-%m-%d")

# Set the 'Date' column as the index (required for time series operations)
df_entertainment.set_index("Date", inplace=True)

def compute_returns(prices_df):
    """
    Given a DataFrame of prices indexed by Date, computes four DataFrames:
      - daily_simple, daily_log, weekly_simple, weekly_log.
    Each return series keeps the index as long as the longest time series,
    representing missing values with NaN.
    """
    # Daily returns; use fill_method=None to avoid the FutureWarning
    daily_simple = prices_df.pct_change(fill_method=None)
    daily_log = np.log(prices_df / prices_df.shift(1))
    
    # Weekly returns: resample to Monday-to-Monday (first available price on each Monday)
    weekly_price = prices_df.resample("W-MON").first()
    weekly_simple = weekly_price.pct_change(fill_method=None)
    weekly_log = np.log(weekly_price / weekly_price.shift(1))
    
    return daily_simple, daily_log, weekly_simple, weekly_log

# Compute return series
daily_simple, daily_log, weekly_simple, weekly_log = compute_returns(df_entertainment)

# Look for missing data
daily_simple.isna()
daily_log.isna()
weekly_simple.isna()
weekly_log.isna()

daily_simple.isna().sum()
daily_log.isna().sum()
weekly_simple.isna().sum()
weekly_log.isna().sum()

# For daily_simple: Count the number of observations per year
daily_counts = daily_simple.groupby(daily_simple.index.year).size()
print("Daily observations per year:")
print(daily_counts)

# For weekly_simple: Count the number of observations per year
weekly_counts = weekly_simple.groupby(weekly_simple.index.year).size()
print("\nWeekly observations per year:")
print(weekly_counts)



###############################################################################
# PART 1: "Descriptive Statistics"
# We investigate the characteristics of the available stocks by computing descriptive statistics.
###############################################################################

def compute_stats(series, effective_period, total_periods=None):
    """
    Helper function to compute basic summary statistics.
    It computes: Mean, Variance, Skewness, Kurtosis, Minimum, and Maximum.
    """
    if len(series) == 0:
        return {k: np.nan for k in ['mean_annual', 'variance_annual', 'skewness', 'kurtosis', 'minimum', 'maximum', 'geometric_annual']}
    
    stats = {
        'mean_annual': series.mean() * effective_period,
        'variance_annual': series.var() * effective_period,
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
        'minimum': series.min(),
        'maximum': series.max()
    }
    
    if total_periods is not None:
        cumulative_return = (1 + series).prod() - 1
        stats['geometric_annual'] = (1 + cumulative_return)**(effective_period / total_periods) - 1
        
    return stats

def summary_stats_simple(returns, effective_period, period_name="Simple"):
    """
    Computes summary statistics for a simple returns DataFrame.
    
    For each column (stock) it computes:
      - Annualized Arithmetic Mean: (sample mean) * effective_period
      - Annualized Geometric Mean: using cumulative return over available periods
      - Annualized Variance: (sample variance) * effective_period
      - Skewness, Kurtosis, Minimum, Maximum.
      
    effective_period: Number of periods in a year (e.g., 252 for daily, 52 for weekly).
    period_name: Used for labeling the output.
    """
    summary = {}
    for col in returns.columns:
        series = returns[col].dropna()
        stats = compute_stats(series, effective_period, len(series))
        summary[col] = {
            f'{period_name} Arithmetic Annualized Mean': stats['mean_annual'],
            f'{period_name} Geometric Annualized Mean': stats['geometric_annual'],
            f'{period_name} Annualized Variance': stats['variance_annual'],
            f'{period_name} Skewness': stats['skewness'],
            f'{period_name} Kurtosis': stats['kurtosis'],
            f'{period_name} Minimum': stats['minimum'],
            f'{period_name} Maximum': stats['maximum']
        }
    return pd.DataFrame(summary).T

def summary_stats_log(returns, effective_period, period_name="Log"):
    """
    Computes summary statistics for a log returns DataFrame.
    
    For each column (stock) it computes:
      - Annualized Mean: (sample mean) * effective_period
      - Annualized Variance: (sample variance) * effective_period
      - Skewness, Kurtosis, Minimum, Maximum.
      
    effective_period: Number of periods in a year (e.g., 252 for daily, 52 for weekly).
    period_name: Used for labeling the output.
    """
    summary = {}
    for col in returns.columns:
        series = returns[col].dropna()
        stats = compute_stats(series, effective_period)
        summary[col] = {
            f'{period_name} Annualized Mean': stats['mean_annual'],
            f'{period_name} Annualized Variance': stats['variance_annual'],
            f'{period_name} Skewness': stats['skewness'],
            f'{period_name} Kurtosis': stats['kurtosis'],
            f'{period_name} Minimum': stats['minimum'],
            f'{period_name} Maximum': stats['maximum']
        }
    return pd.DataFrame(summary).T

# Effective periods: use 260 days for daily data (260=52*5) and 52 weeks for weekly data.
effective_days_per_year = 260
effective_weeks_per_year = 52

# Compute summary statistics for each returns DataFrame.
daily_simple_summary = summary_stats_simple(daily_simple, effective_days_per_year, "Daily Simple")
daily_log_summary = summary_stats_log(daily_log, effective_days_per_year, "Daily Log")
weekly_simple_summary = summary_stats_simple(weekly_simple, effective_weeks_per_year, "Weekly Simple")
weekly_log_summary = summary_stats_log(weekly_log, effective_weeks_per_year, "Weekly Log")

# Display summary statistics
print("\nDaily Simple Returns Summary:")
print(daily_simple_summary)
print("\nDaily Log Returns Summary:")
print(daily_log_summary)
print("\nWeekly Simple Returns Summary:")
print(weekly_simple_summary)
print("\nWeekly Log Returns Summary:")
print(weekly_log_summary)



###############################################################################
# PART 2: "Stationarity"
# We test the stationarity of log prices for each asset.
###############################################################################

# -----------------------------------------------------------------------------
# PART 2.0 :
# compute log price time series
# summary statistics
# line graph of log price time series (all companies one graph)
# -----------------------------------------------------------------------------

# Transform the price series into log prices
log_prices = np.log(df_entertainment)

# Summary statistics for log prices
def summary_stats_log_prices(prices_df):
    """
    Computes summary statistics for the log prices.
    For each company it computes: Mean, Variance, Skewness, Kurtosis, Minimum, and Maximum.
    """
    # Using pandas describe() and agg() methods for vectorized operations
    summary = pd.DataFrame({
        'Mean': prices_df.mean(),
        'Variance': prices_df.var(),
        'St. Deviation': prices_df.std(),
        'Skewness': prices_df.skew(),
        'Kurtosis': prices_df.kurtosis(),
        'Minimum': prices_df.min(),
        'Maximum': prices_df.max()
    }).T.T  # Double transpose to maintain the same structure as before

    return summary

log_prices_summary = summary_stats_log_prices(log_prices)
print("\nLog Prices Summary Statistics:")
print(log_prices_summary)

# Line Graph of Log Prices Time Series (all companies on one graph)
plot_time_series(log_prices, "Log Prices Time Series") 
# plot suggests presence of a trend in Log prices


# -----------------------------------------------------------------------------
# PART 2.1 : CRITICAL VALUES
# we compute our own critical values for DF test for our sample size using Monte-Carlo simulations.
# -----------------------------------------------------------------------------

def simulate_df_distribution(T, N, phi=1.0):
    """
    Simulate the distribution of the Dickey-Fuller test statistic for log prices.
    
    Parameters:
      T (int): Length of each simulated time series.
      N (int): Number of Monte Carlo replications.
      phi (float): AR(1) coefficient for the log price process.
                 Use phi=1.0 for a random walk (unit root).
    
    Returns:
      numpy.ndarray: Array of N Dickey-Fuller test statistics computed as t(φ̂ - 1).
    """
    test_stats = np.zeros(N)

    for i in range(N):
        # 1. Simulate error terms εt ~ N(0, 1)
        errors = np.random.normal(0, 1, T)

        # 2. Generate the log price series: 
        #    For phi=1, use a random walk; otherwise simulate an AR(1) process.
        if phi == 1:
            log_prices_sim = np.cumsum(errors)
        else:
            log_prices_sim = np.zeros(T)
            log_prices_sim[0] = errors[0]
            for t in range(1, T):
                log_prices_sim[t] = phi * log_prices_sim[t-1] + errors[t]
        
        # 3. Prepare data for the AR(1) regression: p_t = μ + φ * p_(t-1) + εt
        #    Use observations t = 2,...,T (dropping the very first to reduce initialization effects)
        X = sm.add_constant(log_prices_sim[:-1])
        y = log_prices_sim[1:]
        X, y = X[1:], y[1:]
        model = sm.OLS(y, X).fit()
        
        # 4. Compute the test statistic: t(φ̂ - 1)
        phi_hat = model.params[1]
        se_phi = model.bse[1]
        test_stats[i] = (phi_hat - 1) / se_phi
    
    return test_stats


# -- NON-Stationary RW Simulation (phi = 1) --
print(len(log_prices['NNDO']))
print(len(log_prices['SONY']))

# Set parameters for the Monte Carlo simulation
N_mc = 10000  # Number of replications
T_mc = 3941    # Length of each simulated series

# -- Random Walk Simulation (phi = 1) --
test_stats_rw = simulate_df_distribution(T_mc, N_mc, phi=1.0)

# Compute empirical critical values (percentiles)
critical_values = np.percentile(test_stats_rw, [1, 5, 10])
print("\nEmpirical Critical Values (T=3941, Random Walk):")
print(f"1%: {critical_values[0]:.3f}")
print(f"5%: {critical_values[1]:.3f}")
print(f"10%: {critical_values[2]:.3f}")

plt.figure(figsize=(10, 6))
# Plot histogram
plt.hist(test_stats_rw, bins=50, density=True, alpha=0.5, color="lightblue", label="Histogram")
# Overlay the KDE plot
sns.kdeplot(test_stats_rw, label="Empirical Distribution", color="blue", lw=2)
plt.axvline(critical_values[0], color="red", linestyle="--", label="1% Critical Value")
plt.axvline(critical_values[1], color="orange", linestyle="--", label="5% Critical Value")
plt.axvline(critical_values[2], color="green", linestyle="--", label="10% Critical Value")
plt.title('Distribution of Dickey-Fuller Test Statistic\n(Random Walk in Log Prices, T=3941)')
plt.xlabel('t(φ - 1)')
plt.ylabel('Density')
plt.legend()
plt.show()


# -- Stationary AR(1) Simulation (phi = 0.2) --
#test_stats_ar = simulate_df_distribution(T_mc, N_mc, phi=0.2)
# Compare the distributions (Random Walk vs. AR(1) with phi=0.2)
#plt.figure(figsize=(9, 6))
# Plot histograms for both distributions
#plt.hist(test_stats_rw, bins=50, density=True, alpha=0.5, color='blue', label='Random Walk (φ=1)')
#plt.hist(test_stats_ar, bins=50, density=True, alpha=0.5, color='red', label='AR(1) (φ=0.2)')
# Add KDE overlays using seaborn
#sns.kdeplot(test_stats_rw, color='blue', lw=2)
#sns.kdeplot(test_stats_ar, color='red', lw=2)
#plt.title('Comparison of Test Statistic Distributions\n(Log Prices)')
#plt.xlabel('t(φ - 1)')
#plt.ylabel('Density')
#plt.legend()
#plt.show()


# -- Simulate for T=500 , as we will need the corresponding critical values later in the project --
T_500 = 500
test_stats_rw_500 = simulate_df_distribution(T_500, N_mc, phi=1.0)

# Plot histogram for T=500
plt.figure(figsize=(9, 6))
plt.hist(test_stats_rw_500, bins=50, density=True, alpha=0.7, color='blue')
plt.title('Distribution of Dickey-Fuller Test Statistic\n(Random Walk in Log Prices, T=500)')
plt.xlabel('t(φ - 1)')
plt.ylabel('Density')
plt.show()

# Compute critical values for T=500
critical_values_500 = np.percentile(test_stats_rw_500, [1, 5, 10])
print("\nCritical Values (T=500):")
print(f"1%: {critical_values_500[0]:.3f}")
print(f"5%: {critical_values_500[1]:.3f}")
print(f"10%: {critical_values_500[2]:.3f}")

plt.figure(figsize=(10, 6))
# Plot histogram
plt.hist(test_stats_rw, bins=50, density=True, alpha=0.5, color="lightblue", label="Histogram")
# Overlay the KDE plot
sns.kdeplot(test_stats_rw_500, label="Empirical Distribution", color="blue", lw=2)
plt.axvline(critical_values_500[0], color="red", linestyle="--", label="1% Critical Value")
plt.axvline(critical_values_500[1], color="orange", linestyle="--", label="5% Critical Value")
plt.axvline(critical_values_500[2], color="green", linestyle="--", label="10% Critical Value")
plt.title('Distribution of Dickey-Fuller Test Statistic\n(Random Walk in Log Prices, T=500)')
plt.xlabel('t(φ - 1)')
plt.ylabel('Density')
plt.legend()
plt.show()

# -----------------------------------------------------------------------------
# PART 2.2 : Testing Non-stationarity
# AR(1) regression for log prices.
# test for stationarity using the Dickey-Fuller tests.
# compute the p-value of DF test using the distribution t(ϕ − 1) made at Q2.5.
# -----------------------------------------------------------------------------

# -- Dickey-Fuller test for stationarity on the log-price series --
stationarity_results_DF_MC = []

for company in log_prices.columns:
    series = log_prices[company].dropna()
    print(f"\n----- Analysis for {company} -----\n")
    
    # 1. AR(1) Regression: p_t = µ + ϕ p_(t-1) + ε_t
    df_reg = pd.DataFrame({"p": series})
    df_reg["p_lag"] = df_reg["p"].shift(1)
    df_reg = df_reg.dropna()
    X = sm.add_constant(df_reg["p_lag"])
    y = df_reg["p"]
    
    model = sm.OLS(y, X).fit()
    print("\nAR(1) Regression Results:")
    print(model.summary())
    
    # Extract coefficient and standard error of φ (p_lag)
    phi_hat = model.params["p_lag"]
    se_phi_hat = model.bse["p_lag"]
    
    # Compute t-stat for H0: φ = 1
    t_stat = (phi_hat - 1) / se_phi_hat
    print(f"\nTest statistic for H0: φ = 1 --> t = {t_stat:.4f}")
    
    # compute empirical one-tailed p-value based on simulated distribution
    p_value_empirical = np.mean(test_stats_rw <= t_stat)
    
    stationarity_results_DF_MC.append({
    'Company': company,
    'DF Test Statistic': t_stat,
    'DF p-value': p_value_empirical,
    'Stationarity': 'Stationary' if p_value_empirical < 0.05 else 'Non-stationary'
    })
    
stationarity_results_DF_MC = pd.DataFrame(stationarity_results_DF_MC)
print("=== Monte Carlo-based DF (DF_MC) Test Results ===")
print("The DF_MC test uses a regression with a constant (and no trend). Its c.v. are derived from a Monte Carlo simulation that provides an empirical distribution for the t-statistic t(ϕ - 1) under the null hypothesis (H0: φ = 1 and µ = 0). The log-price series is simulated as a simple random walk (without drift), therefore we are assuming that the true process under the null is a random walk! from which we get the null hypothesis H0: φ = 1 conditional upon the maintained assumption that µ = 0. This empirical distribution can yield more accurate inference in finite samples.\n")
print(stationarity_results_DF_MC)


###############################################################################
# PART 3: "Cointegration"
# We test for cointegration for all the possible pairs in our dataset.
###############################################################################

def cointegration_df_statistic(residuals):
    """
    Given the residuals z_t from a cointegration regression,
    compute the Dickey-Fuller test statistic by estimating:
       Δz_t = μ + φ·z_{t-1} + ε_t.
    Returns the t-statistic for the coefficient φ.
    """
    # Convert to numpy array if residuals is a pandas Series.
    z = residuals.values if hasattr(residuals, "values") else residuals
    # Compute differences: Δz_t = z_t - z_{t-1}
    dz = np.diff(z)
    # Use z_{t-1} as the lagged residual (matching the length of dz)
    z_lag = z[:-1]
    X = sm.add_constant(z_lag)
    model = sm.OLS(dz, X).fit()
    t_stat = model.tvalues[1]  # t-value for the coefficient on z_lag
    return t_stat


def simulate_cointegration_df_distribution(T, N, seed=None):
    """
    Simulate the distribution of the Dickey-Fuller test statistic for cointegration testing.
    
    Procedure:
      1. Simulate two independent random walks p_A and p_B of length T.
      2. Estimate the cointegration regression: p_A = α + β·p_B + z.
      3. Compute the DF test statistic on the residuals z by regressing:
             Δz_t = μ + φ·z_{t-1} + ε_t
         and obtaining the t-statistic for φ.
      4. Repeat N times.
    
    Returns:
      A numpy array of N simulated t-statistics.
    """
    if seed is not None:
        np.random.seed(seed)
    t_stats = np.zeros(N)
    for i in range(N):
        # Simulate independent random walks
        errors_A = np.random.normal(0, 1, T)
        errors_B = np.random.normal(0, 1, T)
        p_A = np.cumsum(errors_A)
        p_B = np.cumsum(errors_B)
        # Estimate cointegration regression: p_A = α + β·p_B + z
        X_sim = sm.add_constant(p_B)
        model_sim = sm.OLS(p_A, X_sim).fit()
        z_sim = model_sim.resid
        # Compute the DF test statistic on the simulated residuals
        t_stats[i] = cointegration_df_statistic(z_sim)
    return t_stats

# -- Simulate for T=500 , as we will need the corresponding critical values later in the project --
T_sim = 500
N_sim=10000
test_stats_rw_500_cointegration =  simulate_cointegration_df_distribution(T_sim, N_sim, seed=42)
# Compute critical values for T=500
critical_values_500_cointegration = np.percentile(test_stats_rw_500_cointegration, [10, 5, 1])
print(f"DF Critical Values for T = {T_sim}:")
print(f"1%: {critical_values_500_cointegration[2]:.3f}")
print(f"5%: {critical_values_500_cointegration[1]:.3f}")
print(f"10%: {critical_values_500_cointegration[0]:.3f}")


# -- Simulate for T=3941 ------------------------------------------------------
T_sim = 3941
N_sim=10000
test_stats_rw_3941_cointegration = simulate_cointegration_df_distribution(T_sim, N_sim, seed=42)
# Compute empirical critical values fot T=3941
critical_values_3941_cointegration = np.percentile(test_stats_rw_3941_cointegration, [10, 5, 1])
print(f"DF Critical Values for T = {T_sim}:")
print(f"1%: {critical_values_3941_cointegration[2]:.3f}")
print(f"5%: {critical_values_3941_cointegration[1]:.3f}")
print(f"10%: {critical_values_3941_cointegration[0]:.3f}")

plt.figure(figsize=(10, 6))
# Plot histogram
plt.hist(test_stats_rw_3941_cointegration, bins=50, density=True, alpha=0.5, color="lightblue", label="Histogram")
# Overlay the KDE plot
sns.kdeplot(test_stats_rw_3941_cointegration, label="Empirical Distribution", color="blue", lw=2)
plt.axvline(critical_values_3941_cointegration[2], color="red", linestyle="--", label="1% Critical Value")
plt.axvline(critical_values_3941_cointegration[1], color="orange", linestyle="--", label="5% Critical Value")
plt.axvline(critical_values_3941_cointegration[0], color="green", linestyle="--", label="10% Critical Value")
plt.title('Distribution of Dickey-Fuller Test Statistic\n(Random Walk in Log Prices, T=3941)')
plt.xlabel('t(φ)')
plt.ylabel('Density')
plt.legend()
plt.show()
#------------------------------------------------------------------------------


def test_cointegration(price_y, price_x, N_sim=10000):
    """
    Test for cointegration between two price series using the Engle-Granger method,
    following the procedure in Section 3.1 of the assignment.
    
    Procedure:
      1. Estimate the cointegration regression:
             price_y = α + β·price_x + z
         and compute the residuals z.
      2. Compute the Dickey-Fuller test statistic on z by estimating:
             Δz_t = μ + φ·z_{t-1} + ε_t
         (this is our DF statistic for cointegration).
      3. Simulate the distribution of t(φ) under the null of no cointegration by
         generating two independent random walks of length T (T = sample size) and
         repeating the cointegration DF procedure N_sim times.
      4. Compute an empirical one-tailed p-value: the proportion of simulated t-statistics
         that are less than or equal to the observed DF statistic.
    
    Returns a dictionary with:
      - 'dependent': name of dependent series
      - 'independent': name of independent series
      - 'test_statistic': the DF test statistic from the residual regression
      - 'p_value': empirical one-tailed p-value based on the simulated distribution
      - 'beta': estimated β from the cointegration regression
      - 'alpha': estimated α from the cointegration regression
      - 'residuals': residual series from the cointegration regression
      - 'cointegrated': True if p_value < 0.05, otherwise False.
    """
    name_y = price_y.name
    name_x = price_x.name

    df = pd.concat([price_y, price_x], axis=1).dropna()
    y = df.iloc[:, 0]
    X = sm.add_constant(df.iloc[:, 1])
    model = sm.OLS(y, X).fit()
    residuals = model.resid

    # Compute the cointegration DF statistic on the residuals.
    df_stat = cointegration_df_statistic(residuals)
    
    #simulated_stats = simulate_cointegration_df_distribution(T_sim, N_sim)
    simulated_stats = test_stats_rw_3941_cointegration
    
    # Compute the one-tailed empirical p-value:
    # Under the null, lower (more negative) values indicate evidence against no cointegration.
    empirical_p_value = np.mean(simulated_stats <= df_stat)
    
    return {
        'dependent': name_y,
        'independent': name_x,
        'test_statistic': df_stat,
        'p_value': empirical_p_value,
        'beta': model.params.iloc[1],
        'alpha': model.params.iloc[0],
        'residuals': residuals,
        'cointegrated': empirical_p_value < 0.05
    }


def find_all_ordered_cointegrated_pairs(log_prices_df, p_value_threshold=0.05):
    """
    Test cointegration for all 20 ordered pairs (5 stocks).
    """
    results = []
    for stock_a, stock_b in permutations(log_prices_df.columns, 2):
        result = test_cointegration(log_prices_df[stock_a], log_prices_df[stock_b])
        results.append(result)
    return pd.DataFrame(results)

def plot_cointegration_residuals(result):
    """
    Plot residuals and log prices from a cointegration regression result.
    """
    y_name = result['dependent']
    x_name = result['independent']
    residuals = result['residuals']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Log prices
    ax1.plot(log_prices[y_name], label=y_name)
    ax1.plot(log_prices[x_name], label=x_name)
    ax1.set_title(f'Log Prices: {y_name} vs {x_name}')
    ax1.set_ylabel("Log Price")
    ax1.legend()

    # Residuals
    ax2.plot(residuals)
    ax2.set_title(f'Residuals from {y_name} ~ {x_name}\nADF Stat: {result["test_statistic"]:.4f}, p-value: {result["p_value"]:.4f}')
    ax2.set_ylabel("Residuals")

    #plt.tight_layout()
    plt.show()

def analyze_cointegration(log_prices_df):
    """
    Analyze and plot cointegration between all ordered pairs.
    Reports the test statistic, p-values, cointegration decision, and
    parameter estimates αˆ and βˆ for each pair.
    Finally, reports which pair is the most strongly cointegrated.
    """
    results_df = find_all_ordered_cointegrated_pairs(log_prices_df)
    
    # Print detailed results for each pair
    print("=== Cointegration Test Results for Each Pair ===\n")
    for idx, row in results_df.iterrows():
        print(f"Dependent: {row['dependent']}, Independent: {row['independent']}")
        print(f" DF Test Statistic: {row['test_statistic']:.4f}")
        print(f" DF p-value: {row['p_value']:.4f}")
        print(f"  Alpha (αˆ): {row['alpha']:.4f}")
        print(f"  Beta  (βˆ): {row['beta']:.4f}")
        print(f"  Cointegrated: {'Yes' if row['cointegrated'] else 'No'}")
        print("-" * 50)
    
    # Report summary table (subset of key columns)
    print("\n=== Summary Table ===")
    summary_cols = ['dependent', 'independent', 'test_statistic', 'p_value', 'alpha', 'alpha_p_value', 'beta', 'beta_p_value', 'cointegrated']
    summary_df = results_df[summary_cols].copy()
    print(summary_df)
    
    # Identify cointegrated pairs
    cointegrated_pairs = results_df[results_df['cointegrated']]
    if not cointegrated_pairs.empty:
        # Most strongly cointegrated: choose the one with the lowest p-value
        strongest = cointegrated_pairs.loc[cointegrated_pairs['p_value'].idxmin()]
        print("\n=== Most Strongly Cointegrated Pair ===")
        print(f"Dependent: {strongest['dependent']}, Independent: {strongest['independent']}")
        print(f" DF Test Statistic: {strongest['test_statistic']:.4f}")
        print(f" DF p-value: {strongest['p_value']:.4f}")
        print(f"  Alpha (αˆ): {strongest['alpha']:.4f}")
        print(f"  Beta  (βˆ): {strongest['beta']:.4f}")
    else:
        print("\nNo cointegrated pairs were found at the 5% significance level.")

    # Plot residuals for each cointegrated pair
    for _, row in cointegrated_pairs.iterrows():
        print(f"\nPlotting residuals for pair: {row['dependent']} ~ {row['independent']}")
        plot_cointegration_residuals(row)
        
    return summary_df

# Run the analysis
cointegration_summary = analyze_cointegration(log_prices)

cointegrated_pair_prices = df_entertainment[['NNDO', 'SONY']]
# Line Graph of Prices Time Series (NNDO and SONY)
plot_time_series(cointegrated_pair_prices, "Prices Time Series") 



###############################################################################
# PART 4: "Pair Trading"
# We aim at exploiting statistical arbitrage underlying the cointegration relationship with a pair-trading strategy
###############################################################################






















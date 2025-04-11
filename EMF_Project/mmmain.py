import os 
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import jarque_bera, norm, gaussian_kde
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey, het_white
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

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

# ---------------------------
# Summary Statistics Functions
# ---------------------------
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
        total_periods = len(series)
        if total_periods == 0:
            # If no data is present, fill with NaN
            arithmetic_annual = np.nan
            geometric_annual = np.nan
            variance_annual = np.nan
            skewness = np.nan
            kurtosis = np.nan
            minimum = np.nan
            maximum = np.nan
        else:
            arithmetic_mean = series.mean()
            arithmetic_annual = arithmetic_mean * effective_period
            variance_annual = series.var() * effective_period
            skewness = series.skew()
            kurtosis = series.kurtosis()
            minimum = series.min()
            maximum = series.max()
            # Compute cumulative return over the sample period
            cumulative_return = (1 + series).prod() - 1
            geometric_annual = (1 + cumulative_return)**(effective_period / total_periods) - 1
            
        summary[col] = {
            f'{period_name} Arithmetic Annualized Mean': arithmetic_annual,
            f'{period_name} Geometric Annualized Mean': geometric_annual,
            f'{period_name} Annualized Variance': variance_annual,
            f'{period_name} Skewness': skewness,
            f'{period_name} Kurtosis': kurtosis,
            f'{period_name} Minimum': minimum,
            f'{period_name} Maximum': maximum
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
        if len(series) == 0:
            mean_annual = np.nan
            variance_annual = np.nan
            skewness = np.nan
            kurtosis = np.nan
            minimum = np.nan
            maximum = np.nan
        else:
            mean_annual = series.mean() * effective_period
            variance_annual = series.var() * effective_period
            skewness = series.skew()
            kurtosis = series.kurtosis()
            minimum = series.min()
            maximum = series.max()
        summary[col] = {
            f'{period_name} Annualized Mean': mean_annual,
            f'{period_name} Annualized Variance': variance_annual,
            f'{period_name} Skewness': skewness,
            f'{period_name} Kurtosis': kurtosis,
            f'{period_name} Minimum': minimum,
            f'{period_name} Maximum': maximum
        }
    return pd.DataFrame(summary).T

# Effective periods: use 252 trading days for daily data and 52 weeks for weekly data.
effective_days_per_year = 252
effective_weeks_per_year = 52

# Compute summary statistics for each returns DataFrame.
daily_simple_summary = summary_stats_simple(daily_simple, effective_days_per_year, "Daily Simple")
daily_log_summary = summary_stats_log(daily_log, effective_days_per_year, "Daily Log")
weekly_simple_summary = summary_stats_simple(weekly_simple, effective_weeks_per_year, "Weekly Simple")
weekly_log_summary = summary_stats_log(weekly_log, effective_weeks_per_year, "Weekly Log")

# ---------------------------
# Display Summary Statistics
# ---------------------------
print("\nDaily Simple Returns Summary:")
print(daily_simple_summary)
print("\nDaily Log Returns Summary:")
print(daily_log_summary)
print("\nWeekly Simple Returns Summary:")
print(weekly_simple_summary)
print("\nWeekly Log Returns Summary:")
print(weekly_log_summary)

# ---------------------------
# Plotting Histograms for Each Company
# ---------------------------
companies = df_entertainment.columns  # ["ATVI", "NNDO", "SONY", "EA", "MSFT"]

# Daily histograms (simple vs log) for each company
for comp in companies:
    fig, ax = plt.subplots(figsize=(10, 6))
    data_simple = daily_simple[comp].dropna()
    data_log = daily_log[comp].dropna()
    ax.hist(data_simple, bins=30, alpha=0.5, label='Daily Simple')
    ax.hist(data_log, bins=30, alpha=0.5, label='Daily Log')
    ax.set_title(f'Daily Returns Distribution for {comp}')
    ax.set_xlabel('Returns')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Weekly histograms (simple vs log) for each company
for comp in companies:
    fig, ax = plt.subplots(figsize=(10, 6))
    data_simple = weekly_simple[comp].dropna()
    data_log = weekly_log[comp].dropna()
    ax.hist(data_simple, bins=20, alpha=0.5, label='Weekly Simple')
    ax.hist(data_log, bins=20, alpha=0.5, label='Weekly Log')
    ax.set_title(f'Weekly Returns Distribution for {comp}')
    ax.set_xlabel('Returns')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# Bar Charts of the Moments for Daily & Weekly (Simple & Log)
# ---------------------------
def plot_bar_chart(summary_df, title):
    """
    Given a summary statistics DataFrame (with companies as rows and moments as columns),
    plots a separate bar chart for each moment across companies.
    """
    moments = summary_df.columns
    companies = summary_df.index
    num_moments = len(moments)
    
    # Create a separate subplot for each moment
    fig, axs = plt.subplots(nrows=num_moments, figsize=(10, 4*num_moments))
    # If only one moment exists, wrap axs in a list
    if num_moments == 1:
        axs = [axs]
    
    for i, moment in enumerate(moments):
        axs[i].bar(companies, summary_df[moment])
        axs[i].set_title(moment)
        axs[i].set_xlabel('Company')
        axs[i].set_ylabel('Value')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Plot bar charts for each summary statistics DataFrame
plot_bar_chart(daily_simple_summary, "Daily Simple Returns Moments")
plot_bar_chart(daily_log_summary, "Daily Log Returns Moments")
plot_bar_chart(weekly_simple_summary, "Weekly Simple Returns Moments")
plot_bar_chart(weekly_log_summary, "Weekly Log Returns Moments")

# ---------------------------
# Line Graphs of Time Series of Returns Over Time
# ---------------------------
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

plot_time_series(daily_simple, "Daily Simple Returns Time Series")
plot_time_series(daily_log, "Daily Log Returns Time Series")
plot_time_series(weekly_simple, "Weekly Simple Returns Time Series")
plot_time_series(weekly_log, "Weekly Log Returns Time Series")

# ---------------------------
# Distribution Plots with Frequency Distribution, KDE, Normal PDF & Jarque-Bera Test
# ---------------------------
def plot_distribution(series, company, return_type, abscissa):
    """
    Plots a distribution for a given returns series (for one company) that includes:
      - Histogram (frequency distribution)
      - Kernel Density Estimate (KDE)
      - Fitted Normal Distribution Curve (using sample mean and std)
      Also performs a Jarque-Bera test for normality and displays the test statistic and p-value.
    """
    plt.figure(figsize=(10,6))
    
    # Plot histogram (frequency distribution)
    plt.hist(series.dropna(), bins=30, density=True, alpha=0.5, label="FD")
    
    # Compute and plot KDE
    kde = gaussian_kde(series.dropna())
    x = np.linspace(series.min(), series.max(), 1000)
    plt.plot(x, kde(x), label="PDF")
    
    # Plot Normal distribution curve with same mean and std dev as series
    mu, sigma = series.mean(), series.std()
    plt.plot(x, norm.pdf(x, mu, sigma), label="Normal")
    
    # Perform Jarque-Bera test for normality
    jb_stat, jb_pvalue = jarque_bera(series.dropna())
    
    plt.title(f"{company} {return_type} Distribution\nJarque-Bera: Stat={jb_stat:.2f}, p-value={jb_pvalue:.3f}")
    plt.xlabel(f"{abscissa}")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# For demonstration, create distribution plots for daily simple returns for each company.
for company in daily_simple.columns:
    plot_distribution(daily_simple[company], company, "Daily Simple Returns", "Returns")
    plot_distribution(daily_log[company], company, "Daily Log Returns", "Returns")
    plot_distribution(weekly_simple[company], company, "Weekly Simple Returns", "Returns")
    plot_distribution(weekly_log[company], company, "Weekly Log Returns", "Returns")


# =============================================================================
# Now, perform the same set of summary statistics, graphs, and tables for the
# log prices time series in df_entertainment.
# =============================================================================

# Transform the price series into log prices.
log_prices = np.log(df_entertainment)

def summary_stats_log_prices(prices_df):
    """
    Computes summary statistics for the log prices.
    For each company it computes: Mean, Variance, Skewness, Kurtosis, Minimum, and Maximum.
    """
    summary = {}
    for col in prices_df.columns:
        series = prices_df[col].dropna()
        if len(series) == 0:
            stats = {
                'Mean': np.nan,
                'Variance': np.nan,
                'St. Deviation': np.nan,
                'Skewness': np.nan,
                'Kurtosis': np.nan,
                'Minimum': np.nan,
                'Maximum': np.nan
            }
        else:
            stats = {
                'Mean': series.mean(),
                'Variance': series.var(),
                'St. Deviation': series.stdev(),
                'Skewness': series.skew(),
                'Kurtosis': series.kurtosis(),
                'Minimum': series.min(),
                'Maximum': series.max()
            }
        summary[col] = stats
    return pd.DataFrame(summary).T

log_prices_summary = summary_stats_log_prices(log_prices)
print("\nLog Prices Summary Statistics:")
print(log_prices_summary)

# Bar Chart for Log Prices Summary Statistics
plot_bar_chart(log_prices_summary, "Log Prices Summary Statistics")

# Line Graph of Log Prices Time Series (all companies on one graph)
plot_time_series(log_prices, "Log Prices Time Series")

# Distribution Plots for Log Prices for Each Company (with FD, PDF, Normal PDF & Jarque-Bera)
for company in log_prices.columns:
    plot_distribution(log_prices[company], company, "Log Prices", "Prices")

# =============================================================================
# Additional Analysis for Log Prices: Lilliefors Test and CDF Comparison
# =============================================================================
def plot_log_price_cdf_comparison(series, company):
    """
    For the log price series of a company:
      - Perform the Lilliefors test for normality.
      - Compute and plot the Empirical CDF vs. the Theoretical Normal CDF.
      - Plot the absolute difference |F*(x) - G(x)| between them.
    """
    data = series.dropna()
    if len(data) == 0:
        print(f"No data for {company}.")
        return
    
    # Estimate parameters
    mu, sigma = data.mean(), data.std()
    
    # Lilliefors test
    lf_stat, lf_p = lilliefors(data, dist='norm')
    
    # Create grid for CDF plots
    x_grid = np.linspace(data.min(), data.max(), 1000)
    # Compute empirical CDF on the grid
    empirical_cdf = np.array([np.mean(data <= x) for x in x_grid])
    # Compute theoretical CDF using the normal distribution with estimated parameters
    theoretical_cdf = norm.cdf(x_grid, loc=mu, scale=sigma)
    # Compute absolute difference
    abs_diff = np.abs(empirical_cdf - theoretical_cdf)
    
    # Plot CDFs and the absolute difference
    fig, axs = plt.subplots(nrows=2, figsize=(10,10))
    
    # Plot Empirical vs Theoretical CDF
    axs[0].plot(x_grid, empirical_cdf, label="Empirical CDF", color='blue')
    axs[0].plot(x_grid, theoretical_cdf, label="Theoretical CDF", color='red', linestyle='--')
    axs[0].set_title(f"{company} Log Prices CDF\nLilliefors Test: Stat={lf_stat:.2f}, p-value={lf_p:.3f}")
    axs[0].set_xlabel("Log Prices")
    axs[0].set_ylabel("CDF")
    axs[0].legend()
    
    # Plot absolute difference between the CDFs
    axs[1].plot(x_grid, abs_diff, label="|Empirical CDF - Theoretical CDF|", color='green')
    axs[1].set_title(f"{company} Absolute Difference between CDFs")
    axs[1].set_xlabel("Log Prices")
    axs[1].set_ylabel("Absolute Difference")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

# Perform the CDF comparison and Lilliefors test for each company's log prices.
for company in log_prices.columns:
    plot_log_price_cdf_comparison(log_prices[company], company)
    
# =============================================================================
# Additional Analysis for Log Prices: Correlogram, Regression & Diagnostics
# =============================================================================
print("\n\n============================")
print("Log Prices Diagnostics")
print("============================\n")

# For each company's log prices series, perform diagnostics.
for company in log_prices.columns:
    series = log_prices[company].dropna()
    print(f"\n----- Analysis for {company} -----\n")
    
    # 1. Correlogram and Ljung-Box test
    plt.figure(figsize=(10, 6))
    plot_acf(series, lags=20)
    plt.title(f"Correlogram for {company} Log Prices")
    plt.show()
    
    lb_test = acorr_ljungbox(series, lags=[10], return_df=True)
    print("Ljung-Box test results:")
    print(lb_test)
    
    # 2. AR(1) Regression: p_t = µ + ϕ p_(t-1) + ε_t
    df_reg = pd.DataFrame({"p": series})
    df_reg["p_lag"] = df_reg["p"].shift(1)
    df_reg = df_reg.dropna()
    X = sm.add_constant(df_reg["p_lag"])
    y = df_reg["p"]
    
    model = sm.OLS(y, X).fit()
    print("\nAR(1) Regression Results:")
    print(model.summary())
    
    # 3. Durbin–Watson test on the regression residuals
    dw = durbin_watson(model.resid)
    print(f"\nDurbin-Watson test for {company}: {dw:.4f}")
    
    # 4. Breusch–Godfrey (Godfrey) test for autocorrelation in regression residuals
    bg_test = acorr_breusch_godfrey(model, nlags=5)
    print(f"Breusch-Godfrey test for {company}: LM stat={bg_test[0]:.4f}, p-value={bg_test[1]:.4f}")
    
    # 5. White's test for heteroskedasticity
    white_test = het_white(model.resid, X)
    print(f"White's heteroskedasticity test for {company}: LM stat={white_test[0]:.4f}, p-value={white_test[1]:.4f}")
    
    # 6. Dickey-Fuller test for stationarity on the log-price series
    adf_result = adfuller(series)
    print(f"\nDickey-Fuller test for {company}:")
    print(f"Test Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    
    # (Optionally, you may also plot the residuals correlogram)
    plt.figure(figsize=(10, 6))
    plot_acf(model.resid, lags=20)
    plt.title(f"Correlogram of Regression Residuals for {company}")
    plt.show()
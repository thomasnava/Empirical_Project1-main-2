import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import os
import sys

# Set working directory to the script's location
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(base_dir)
sys.path.insert(0, base_dir)

def simulate_df_distribution(T, N, phi=1.0):
    """
    Simulate Dickey-Fuller test statistic distribution for log prices
    
    Parameters:
    T (int): Length of each time series
    N (int): Number of Monte Carlo replications
    phi (float): AR(1) coefficient for log price process
    
    Returns:
    numpy array: Array of N test statistics
    """
    test_stats = np.zeros(N)
    
    for i in range(N):
        # 1. Simulate error terms εt ~ N(0, 1)
        errors = np.random.normal(0, 1, T)
        
        # 2. Generate log price series pt = μ + φpt-1 + εt
        log_prices = np.zeros(T)
        log_prices[0] = errors[0]  # Initialize first value
        
        # Generate log prices using AR(1) process
        for t in range(1, T):
            log_prices[t] = phi * log_prices[t-1] + errors[t]
        
        # 3. Estimate AR(1) model for log prices
        X = sm.add_constant(log_prices[:-1])  # Add constant and use lagged log prices
        y = log_prices[1:]  # Dependent variable is current log prices
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # 4. Compute test statistic t(φ - 1)
        phi_hat = model.params[1]  # AR coefficient is second parameter (after constant)
        se_phi = model.bse[1]  # Standard error of AR coefficient
        test_stats[i] = (phi_hat - 1) / se_phi
    
    return test_stats

def main():
    # File paths from import os.py
    input_file = os.path.join("Data", "Data_Project1.xlsx")
    output_file = os.path.join("Data", "Entertainment_data.xlsx")

    # Load the data as in import os.py
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
    else:
        # Load the existing file
        df_entertainment = pd.read_excel(output_file, sheet_name="Entertainment")
        df_entertainment["Date"] = pd.to_datetime(df_entertainment["Date"], format="%Y-%m-%d")

    # Set the 'Date' column as the index
    df_entertainment.set_index("Date", inplace=True)

    # Transform to log prices as in import os.py
    log_prices = np.log(df_entertainment)

    # Set parameters for Monte Carlo simulation
    N = 10000  # Number of Monte Carlo replications
    T = 100  # Sample size from Fuller (1976)

    # Simulate for random walk in log prices (phi=1)
    test_stats_rw = simulate_df_distribution(T, N, phi=1.0)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(test_stats_rw, bins=50, density=True, alpha=0.7, color='blue')
    plt.title('Distribution of Dickey-Fuller Test Statistic\n(Random Walk in Log Prices)')
    plt.xlabel('t(φ - 1)')
    plt.ylabel('Density')

    # Add normal distribution for comparison
    x = np.linspace(min(test_stats_rw), max(test_stats_rw), 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r--', label='Standard Normal')
    plt.legend()
    plt.show()

    # Compute critical values
    critical_values = np.percentile(test_stats_rw, [1, 5, 10])
    print("\nCritical Values (T=100):")
    print(f"1%: {critical_values[0]:.3f}")
    print(f"5%: {critical_values[1]:.3f}")
    print(f"10%: {critical_values[2]:.3f}")

    # Simulate for stationary AR(1) in log prices with phi=0.2
    test_stats_ar = simulate_df_distribution(T, N, phi=0.2)

    # Plot comparison of distributions
    plt.figure(figsize=(10, 6))
    plt.hist(test_stats_rw, bins=50, density=True, alpha=0.5, color='blue', label='Random Walk')
    plt.hist(test_stats_ar, bins=50, density=True, alpha=0.5, color='red', label='AR(1) φ=0.2')
    plt.title('Comparison of Test Statistic Distributions\n(Log Prices)')
    plt.xlabel('t(φ - 1)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Now simulate for T=500
    T_500 = 500
    test_stats_rw_500 = simulate_df_distribution(T_500, N, phi=1.0)

    # Plot histogram for T=500
    plt.figure(figsize=(10, 6))
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

if __name__ == "__main__":
    main() 
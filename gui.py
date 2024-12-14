import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, poisson, expon, chi2, beta, gamma, lognorm, weibull_min, invgamma, kstest

"""
Streamlit-based app to compare a given dataset to different distribution types visually and statistically using K-S tests.
"""

def compare_with_normal(data, ax):
    """
    Compare the given data to a Normal distribution.

    Parameters:
    data (array-like): The input data to compare.
    ax (matplotlib axis): Axis to plot the histogram and PDF.
    """
    mean = np.mean(data)
    std_dev = np.std(data)

    ax.hist(data, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(min(data), max(data), 100)
    ax.plot(x, norm.pdf(x, mean, std_dev), color='red', label="Normal Distribution")
    ax.set_title("Comparison with Normal Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

def compare_with_poisson(data, ax):
    lam = np.mean(data)
    ax.hist(data, bins=range(int(min(data)), int(max(data))+1), density=True, color='orange', edgecolor='black', alpha=0.7, label="Data")
    x = np.arange(int(min(data)), int(max(data))+1)
    ax.plot(x, poisson.pmf(x, lam), color='red', marker='o', linestyle='dashed', label="Poisson Distribution")
    ax.set_title("Comparison with Poisson Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    
def compare_with_chi_square(data, ax):
    degrees = int(np.mean(data))
    ax.hist(data, bins=50, density=True, color='salmon', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(0, max(data), 100)
    ax.plot(x, chi2.pdf(x, degrees), color='red', label="Chi-Square Distribution")
    ax.set_title("Comparison with Chi-Square Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    
def compare_with_beta(data, ax):
    alpha, beta_val = 2, 5
    ax.hist(data, bins=50, density=True, color='purple', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(0, 1, 100)
    ax.plot(x, beta.pdf(x, alpha, beta_val), color='red', label="Beta Distribution")
    ax.set_title("Comparison with Beta Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

def compare_with_gamma(data, ax):
    shape = 2
    scale = np.mean(data) / shape
    ax.hist(data, bins=50, density=True, color='gold', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(0, max(data), 100)
    ax.plot(x, gamma.pdf(x, shape, scale=scale), color='red', label="Gamma Distribution")
    ax.set_title("Comparison with Gamma Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    
def compare_with_log_normal(data, ax):
    mean = np.mean(np.log(data[data > 0]))
    std_dev = np.std(np.log(data[data > 0]))
    ax.hist(data, bins=50, density=True, color='teal', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(min(data), max(data), 100)
    ax.plot(x, lognorm.pdf(x, s=std_dev, scale=np.exp(mean)), color='red', label="Log-Normal Distribution")
    ax.set_title("Comparison with Log-Normal Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

def compare_with_weibull(data, ax):
    shape = 1.5
    ax.hist(data, bins=50, density=True, color='brown', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(0, max(data), 100)
    ax.plot(x, weibull_min.pdf(x, shape), color='red', label="Weibull Distribution")
    ax.set_title("Comparison with Weibull Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

def compare_with_inverse_gamma(data, ax):
    shape = 3  # Adjust shape as needed
    scale = np.mean(data) * (shape - 1)
    ax.hist(data, bins=50, density=True, color='green', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(0.01, max(data), 100)
    ax.plot(x, invgamma.pdf(x, a=shape, scale=scale), color='red', label="Inverse Gamma Distribution")
    ax.set_title("Comparison with Inverse Gamma Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

def perform_ks_tests(data):
    """
    Perform Kolmogorov-Smirnov (K-S) tests for the given data against multiple distributions.

    Parameters:
    data (array-like): The input data to test.

    Returns:
    list: Sorted results of the K-S test as tuples of (distribution name, K-S statistic).
    """
    distributions = {
        "Normal": ("norm", [np.mean(data), np.std(data)]),
        "Inverse Gamma": ("invgamma", [3, 0, np.mean(data) * 2]),
        "Poisson": ("poisson", [np.mean(data)]),
        "Chi-Square": ("chi2", [int(np.mean(data))]),
        "Beta": ("beta", [2, 5, 0, 1]),
        "Gamma": ("gamma", [2, 0, np.mean(data) / 2]),
        "Log-Normal": ("lognorm", [np.std(np.log(data[data > 0])), 0, np.exp(np.mean(np.log(data[data > 0])))]),
        "Weibull": ("weibull_min", [1.5, 0, max(data)])
    }

    ks_results = {}
    for dist_name, (dist, params) in distributions.items():
        ks_stat, p_value = kstest(data, dist, args=tuple(params))
        ks_results[dist_name] = ks_stat

    sorted_results = sorted(ks_results.items(), key=lambda x: x[1])
    return sorted_results

def plot_selected_distribution(data, dist_name, ax):
    if dist_name == "Normal":
        compare_with_normal(data, ax)
    elif dist_name == "Poisson":
        compare_with_poisson(data, ax)
    elif dist_name == "Chi-Square":
        compare_with_chi_square(data, ax)
    elif dist_name == "Beta":
        compare_with_beta(data, ax)
    elif dist_name == "Gamma":
        compare_with_gamma(data, ax)
    elif dist_name == "Log-Normal":
        compare_with_log_normal(data, ax)
    elif dist_name == "Weibull":
        compare_with_weibull(data, ax)
    elif dist_name == "Inverse Gamma":
        compare_with_inverse_gamma(data, ax)

# Streamlit UI
st.title("Distribution Analysis")

# Input data using file uploader
uploaded_file = st.file_uploader("Upload CSV file containing data:", type=["csv"])

# Dropdown for distribution selection
distributions = ["Normal", "Poisson", "Chi-Square", "Beta", "Gamma", "Log-Normal", "Weibull", "Inverse Gamma"]
selected_distribution = st.selectbox("Select distribution:", distributions)

if uploaded_file is not None:
    try:
        # Read CSV and process data
        data_df = pd.read_csv(uploaded_file)
        column_name = st.selectbox("Select column:", data_df.columns)
        data = data_df[column_name].dropna().to_numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot selected distribution
        plot_selected_distribution(data, selected_distribution, ax)

        # Perform K-S tests
        ks_results = perform_ks_tests(data)
        best_fit = ks_results[0]

        # Display results
        st.pyplot(fig)
        st.markdown(f"**Best Fit:** {best_fit[0]}  ")
        st.markdown(f"**K-S Statistic:** {best_fit[1]:.4f}")

        # Output all K-S test results
        st.markdown("### K-S Test Results for All Distributions:")
        for dist_name, ks_stat in ks_results:
            st.markdown(f"- **{dist_name}:** K-S Statistic = {ks_stat:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")

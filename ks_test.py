# Filename: correlation_generator.py

import numpy as np
from scipy.stats import kstest, beta, norm, gamma, lognorm
import argparse

def calculate_shape_similarity_with_ks(data, dist_name, *params):
    """
    Perform the Kolmogorov-Smirnov test to compare the shape of data with the specified distribution.

    Parameters:
    data (array-like): The data to be tested.
    dist_name (str): The name of the distribution to compare with.
    *params: The parameters for the distribution.

    Returns:
    float: The p-value of the KS test, indicating the similarity of data with the distribution.
    """
    stat, p_val = kstest(data, dist_name, args=params)
    return p_val

def calculate_shape_similarity_with_beta(data):
    """
    Calculate shape similarity for the Beta distribution.

    Parameters:
    data (array-like): The data to be tested.

    Returns:
    float: The p-value of the KS test for Beta distribution.
    """
    alpha, beta_val = 2, 5
    return calculate_shape_similarity_with_ks(data, 'beta', alpha, beta_val)

def calculate_shape_similarity_with_normal(data):
    """
    Calculate shape similarity for the Normal distribution.

    Parameters:
    data (array-like): The data to be tested.

    Returns:
    float: The p-value of the KS test for Normal distribution.
    """
    mean, std_dev = np.mean(data), np.std(data)
    return calculate_shape_similarity_with_ks(data, 'norm', mean, std_dev)

def calculate_shape_similarity_with_gamma(data):
    """
    Calculate shape similarity for the Gamma distribution.

    Parameters:
    data (array-like): The data to be tested.

    Returns:
    float: The p-value of the KS test for Gamma distribution.
    """
    shape, loc, scale = gamma.fit(data)
    return calculate_shape_similarity_with_ks(data, 'gamma', shape, loc, scale)

def calculate_shape_similarity_with_log_normal(data):
    """
    Calculate shape similarity for the Log-Normal distribution.

    Parameters:
    data (array-like): The data to be tested.

    Returns:
    float: The p-value of the KS test for Log-Normal distribution.
    """
    data = data[data > 0]

    # Fit the log-normal distribution parameters
    shape, loc, scale = lognorm.fit(data, floc=0)  # Often, loc is fixed to 0 for pure log-normal

    return calculate_shape_similarity_with_ks(data, 'lognorm', shape, loc, scale)

def main():
    parser = argparse.ArgumentParser(description="Calculate shape similarity using Kolmogorov-Smirnov test.")
    parser.add_argument("datafile", type=str, help="Path to the file containing the dataset (in CSV format).")

    args = parser.parse_args()

    # Load the dataset
    try:
        data = np.loadtxt(args.datafile, delimiter=",")
    except Exception as e:
        print(f"Error loading data file: {e}")
        return

    # Calculate similarity scores
    similarity_scores = {
        "Normal": calculate_shape_similarity_with_normal(data),
        "Beta": calculate_shape_similarity_with_beta(data),
        "Gamma": calculate_shape_similarity_with_gamma(data),
        "Log-Normal": calculate_shape_similarity_with_log_normal(data),
    }

    print("=" * 50)
    print("\nTesting Input Distribution data:")
    for dist, score in similarity_scores.items():
        print(f"Shape similarity with {dist} Distribution (KS p-value): {score:.4f}")
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()

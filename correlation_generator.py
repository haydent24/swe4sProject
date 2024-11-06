import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, beta, norm, gamma, lognorm, weibull_min

"""
Contains functions to compare a given dataset with various distribution types using the KS test and plot the simulated data
"""

def calculate_shape_similarity_with_ks(data, dist_name, *params):
    # Perform the Kolmogorov-Smirnov test to compare the shape of data with the specified distribution
    stat, p_val = kstest(data, dist_name, args=params)
    return p_val  # Higher p-value indicates a better shape match

def calculate_shape_similarity_with_beta(data):
    alpha, beta_val = 2, 5
    simulated_data = np.random.beta(alpha, beta_val, len(data))
    plot_data(simulated_data, "Simulated Beta Distribution")
    return calculate_shape_similarity_with_ks(simulated_data, 'beta', alpha, beta_val)

def calculate_shape_similarity_with_normal(data):
    mean, std_dev = np.mean(data), np.std(data)
    simulated_data = np.random.normal(mean, std_dev, len(data))
    plot_data(simulated_data, "Simulated Normal Distribution")
    return calculate_shape_similarity_with_ks(simulated_data, 'norm', mean, std_dev)

def calculate_shape_similarity_with_gamma(data):
    shape = 2
    scale = np.mean(data) / 2
    
    # Ensure the scale is positive for the Gamma distribution
    if scale <= 0:
        scale = np.abs(scale) + 1e-5  # Ensure scale is positive by taking absolute value and adding a small offset

    simulated_data = np.random.gamma(shape, scale, len(data))
    plot_data(simulated_data, "Simulated Gamma Distribution")
    return calculate_shape_similarity_with_ks(simulated_data, 'gamma', shape, scale)

def calculate_shape_similarity_with_log_normal(data):
    mean, std_dev = np.mean(np.log(data[data > 0])), np.std(np.log(data[data > 0]))
    simulated_data = np.random.lognormal(mean, std_dev, len(data))
    plot_data(simulated_data, "Simulated Log-Normal Distribution")
    return calculate_shape_similarity_with_ks(simulated_data, 'lognorm', std_dev, np.exp(mean))

def calculate_shape_similarity_with_weibull(data):
    shape = 1.5
    simulated_data = np.random.weibull(shape, len(data))
    plot_data(simulated_data, "Simulated Weibull Distribution")
    return calculate_shape_similarity_with_ks(simulated_data, 'weibull_min', shape)

def plot_data(data, title):
    # Plot the histogram of the simulated data
    plt.hist(data, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()  # Show the plot

def main(data):
    similarity_scores = {
        "Normal": calculate_shape_similarity_with_normal(data),
        "Beta": calculate_shape_similarity_with_beta(data),
        "Gamma": calculate_shape_similarity_with_gamma(data),
        "Log-Normal": calculate_shape_similarity_with_log_normal(data),
        "Weibull": calculate_shape_similarity_with_weibull(data),
    }
    
    # Print out the similarity (p-value) for each distribution
    for dist, score in similarity_scores.items():
        print(f"Shape similarity with {dist} Distribution (p-value): {score:.4f}")

if __name__ == '__main__':
    # Example 1: Normal distribution (Test case for Normal distribution)
    data_normal = np.random.normal(0, 1, size=1000)  # Generated from Normal distribution
    print("Testing Normal Distribution data:")
    main(data_normal)
    print("\n" + "="*50 + "\n")

    # Example 2: Beta distribution (Test case for Beta distribution)
    data_beta = np.random.beta(2, 5, size=1000)  # Generated from Beta distribution
    print("Testing Beta Distribution data:")
    main(data_beta)
    print("\n" + "="*50 + "\n")

    # Example 3: Gamma distribution (Test case for Gamma distribution)
    data_gamma = np.random.gamma(2, 2, size=1000)  # Generated from Gamma distribution
    print("Testing Gamma Distribution data:")
    main(data_gamma)
    print("\n" + "="*50 + "\n")

    # Example 4: Log-Normal distribution (Test case for Log-Normal distribution)
    data_log_normal = np.random.lognormal(0, 1, size=1000)  # Generated from Log-Normal distribution
    print("Testing Log-Normal Distribution data:")
    main(data_log_normal)
    print("\n" + "="*50 + "\n")

    # Example 5: Weibull distribution (Test case for Weibull distribution)
    data_weibull = np.random.weibull(1.5, size=1000)  # Generated from Weibull distribution
    print("Testing Weibull Distribution data:")
    main(data_weibull)

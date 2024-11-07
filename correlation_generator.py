import numpy as np
from scipy.stats import kstest, beta, norm, gamma, lognorm, weibull_min

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
    shape = 2
    scale = np.mean(data) / 2
    return calculate_shape_similarity_with_ks(data, 'gamma', shape, scale)

def calculate_shape_similarity_with_log_normal(data):
    """
    Calculate shape similarity for the Log-Normal distribution.
    
    Parameters:
    data (array-like): The data to be tested.
    
    Returns:
    float: The p-value of the KS test for Log-Normal distribution.
    """
    mean, std_dev = np.mean(np.log(data[data > 0])), np.std(np.log(data[data > 0]))
    return calculate_shape_similarity_with_ks(data, 'lognorm', std_dev, np.exp(mean))

def calculate_shape_similarity_with_weibull(data):
    """
    Calculate shape similarity for the Weibull distribution.
    
    Parameters:
    data (array-like): The data to be tested.
    
    Returns:
    float: The p-value of the KS test for Weibull distribution.
    """
    shape = 1.5
    return calculate_shape_similarity_with_ks(data, 'weibull_min', shape)

def main():
    # Test Data
    data_sets = {
        "Normal": np.random.normal(0, 1, size=1000),
        "Beta": np.random.beta(2, 5, size=1000),
        "Gamma": np.random.gamma(2, 2, size=1000),
        "Log-Normal": np.random.lognormal(0, 1, size=1000),
        "Weibull": np.random.weibull(1.5, size=1000)
    }

    # Initialize summary dictionary
    similarity_scores_summary = {}

    # For each dataset, calculate p-value for each distribution
    for dist_name, data in data_sets.items():
        print(f"Testing {dist_name} Distribution data:")

        similarity_scores = {
            "Normal": calculate_shape_similarity_with_normal(data),
            "Beta": calculate_shape_similarity_with_beta(data),
            "Gamma": calculate_shape_similarity_with_gamma(data),
            "Log-Normal": calculate_shape_similarity_with_log_normal(data),
            "Weibull": calculate_shape_similarity_with_weibull(data),
        }

        # Print similarity scores for each distribution
        for dist, score in similarity_scores.items():
            print(f"Shape similarity with {dist} Distribution (KS p-value): {score:.4f}")
        
        # Add summary of scores for each distribution to dictionary
        similarity_scores_summary[dist_name] = similarity_scores

        print("\n" + "="*50 + "\n")

if __name__ == '__main__':
    main()

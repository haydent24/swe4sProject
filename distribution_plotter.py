import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.stats import norm, poisson, expon, chi2, beta, gamma, lognorm, weibull_min

"""
Contains functions to compare a given dataset to different distribution types visually
"""

def compare_with_normal(data):
    mean = np.mean(data)
    std_dev = np.std(data)

    plt.hist(data, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, norm.pdf(x, mean, std_dev), color='red', label="Normal Distribution")
    plt.title("Comparison with Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def compare_with_poisson(data):
    lam = np.mean(data)
    plt.hist(data, bins=range(int(min(data)), int(max(data))+1), density=True, color='orange', edgecolor='black', alpha=0.7, label="Data")
    x = np.arange(int(min(data)), int(max(data))+1)
    plt.plot(x, poisson.pmf(x, lam), color='red', marker='o', linestyle='dashed', label="Poisson Distribution")
    plt.title("Comparison with Poisson Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    
def compare_with_chi_square(data):
    degrees = int(np.mean(data))
    plt.hist(data, bins=50, density=True, color='salmon', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(0, max(data), 100)
    plt.plot(x, chi2.pdf(x, degrees), color='red', label="Chi-Square Distribution")
    plt.title("Comparison with Chi-Square Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    
def compare_with_beta(data):
    alpha, beta_val = 2, 5
    plt.hist(data, bins=50, density=True, color='purple', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(0, 1, 100)
    plt.plot(x, beta.pdf(x, alpha, beta_val), color='red', label="Beta Distribution")
    plt.title("Comparison with Beta Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def compare_with_gamma(data):
    shape = 2
    scale = np.mean(data) / shape
    plt.hist(data, bins=50, density=True, color='gold', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(0, max(data), 100)
    plt.plot(x, gamma.pdf(x, shape, scale=scale), color='red', label="Gamma Distribution")
    plt.title("Comparison with Gamma Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    
def compare_with_log_normal(data):
    mean = np.mean(np.log(data[data > 0]))
    std_dev = np.std(np.log(data[data > 0]))
    plt.hist(data, bins=50, density=True, color='teal', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, lognorm.pdf(x, s=std_dev, scale=np.exp(mean)), color='red', label="Log-Normal Distribution")
    plt.title("Comparison with Log-Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def compare_with_weibull(data):
    shape = 1.5
    plt.hist(data, bins=50, density=True, color='brown', edgecolor='black', alpha=0.7, label="Data")
    x = np.linspace(0, max(data), 100)
    plt.plot(x, weibull_min.pdf(x, shape), color='red', label="Weibull Distribution")
    plt.title("Comparison with Weibull Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def main(data):
    print("Comparing dataset with Normal Distribution")
    compare_with_normal(data)
    
    print("Comparing dataset with Poisson Distribution")
    compare_with_poisson(data)
    
    print("Comparing dataset with Chi-Square Distribution")
    compare_with_chi_square(data)
    
    print("Comparing dataset with Beta Distribution")
    compare_with_beta(data)
    
    print("Comparing dataset with Gamma Distribution")
    compare_with_gamma(data)
    
    print("Comparing dataset with Log-Normal Distribution")
    compare_with_log_normal(data)
    
    print("Comparing dataset with Weibull Distribution")
    compare_with_weibull(data)

if __name__ == '__main__':
    # Replace with actual dataset or generate sample data for testing
    # Example: data = np.random.normal(0, 1, size=1000)
    data = np.random.normal(0, 1, size=1000)  # Example dataset
    main(data)

import numpy as np
import pandas as pd
import os

# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Ensure that the 'data' folder exists in the same directory as the script
data_folder = os.path.join(current_directory, 'data')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Parameters for each distribution
num_points = 1000  # Number of data points

# Gaussian distribution parameters
mean = 0  # Mean of the Gaussian distribution
std_dev = 1  # Standard deviation of the Gaussian distribution

# Uniform distribution parameters
lower_uniform = 0  # Lower bound of the Uniform distribution
upper_uniform = 1  # Upper bound of the Uniform distribution

# Exponential distribution parameters
lambda_exp = 1  # Rate parameter for the Exponential distribution

# Gamma distribution parameters
shape_gamma = 2  # Shape parameter of the Gamma distribution
scale_gamma = 2  # Scale parameter of the Gamma distribution

# Inverse Gamma distribution parameters
shape_inv_gamma = 3  # Shape parameter of the Inverse Gamma distribution
scale_inv_gamma = 2  # Scale parameter of the Inverse Gamma distribution

# Lognormal distribution parameters
mean_lognormal = 0  # Mean of the Lognormal distribution
std_dev_lognormal = 1  # Standard deviation of the Lognormal distribution

# Function to generate and save each distribution
def generate_and_save_distribution(distribution_name, data, file_name):
    # Create a DataFrame and save to CSV
    df = pd.DataFrame({'Value': data})
    output_file = os.path.join(data_folder, file_name)
    df.to_csv(output_file, index=False)
    print(f"CSV file '{output_file}' has been generated.")

# Gaussian distribution
gaussian_data = np.random.normal(loc=mean, scale=std_dev, size=num_points)
generate_and_save_distribution('Gaussian', gaussian_data, 'gaussian_data.csv')

# Uniform distribution
uniform_data = np.random.uniform(low=lower_uniform, high=upper_uniform, size=num_points)
generate_and_save_distribution('Uniform', uniform_data, 'uniform_data.csv')

# Exponential distribution
exponential_data = np.random.exponential(scale=1/lambda_exp, size=num_points)
generate_and_save_distribution('Exponential', exponential_data, 'exponential_data.csv')

# Gamma distribution
gamma_data = np.random.gamma(shape=shape_gamma, scale=scale_gamma, size=num_points)
generate_and_save_distribution('Gamma', gamma_data, 'gamma_data.csv')

# Inverse Gamma distribution
inv_gamma_data = np.random.gamma(shape=shape_inv_gamma, scale=scale_inv_gamma, size=num_points)
inv_gamma_data = 1 / inv_gamma_data  # Inverse of Gamma for Inverse Gamma distribution
generate_and_save_distribution('Inverse Gamma', inv_gamma_data, 'inverse_gamma_data.csv')

# Lognormal distribution
lognormal_data = np.random.lognormal(mean=mean_lognormal, sigma=std_dev_lognormal, size=num_points)
generate_and_save_distribution('Lognormal', lognormal_data, 'lognormal_data.csv')

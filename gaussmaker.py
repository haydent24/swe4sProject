import numpy as np
import pandas as pd

# Parameters for the Gaussian distribution
mean = 0  # Mean of the distribution
std_dev = 1  # Standard deviation of the distribution
num_samples = 1000  # Number of samples to generate

# Generate random data
data = np.random.normal(loc=mean, scale=std_dev, size=num_samples)

# Create a DataFrame
df = pd.DataFrame({'Gaussian Data': data})

# Save to a CSV file
output_file = 'gaussian_distribution.csv'
df.to_csv(output_file, index=False)

print(f"CSV file '{output_file}' has been generated with {num_samples} samples.")

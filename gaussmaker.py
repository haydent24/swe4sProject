import numpy as np
import pandas as pd

# Parameters for the Gaussian curve
mean = 0  # Mean of the distribution
std_dev = 1  # Standard deviation of the distribution
num_points = 1000  # Number of points to plot

# Generate x values evenly spaced around the meanS
x_values = np.linspace(mean - 4*std_dev, mean + 4*std_dev, num_points)

# Calculate the y values (PDF of the Gaussian distribution)
y_values = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std_dev)**2)

# Create a DataFrame
df = pd.DataFrame({'X': x_values, 'Y': y_values})

# Save to a CSV file
output_file = 'gaussian_curve.csv'
df.to_csv(output_file, index=False)

print(f"CSV file '{output_file}' has been generated.")

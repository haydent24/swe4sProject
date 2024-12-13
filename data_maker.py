"""
This script generates synthetic datasets based on various statistical distributions and saves
them as CSV files. Each distribution has its parameters defined, and the data is saved in
a 'data' directory located in the script's directory. Supported distributions include:
Gaussian, Uniform, Beta, Gamma, Inverse Gamma, and Lognormal.

Functions:
    - generate_distribution: Generates data for a specified distribution.
    - save_distribution: Saves generated data to a CSV file.
    - main: Coordinates the data generation and saving process.

Usage:
    Run this script directly to generate datasets for all predefined distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Define the number of data points for each distribution
NUM_POINTS = 1000

# Define distribution parameters
DISTRIBUTION_PARAMS = {
    "Gaussian": {"loc": 0, "scale": 1},
    "Uniform": {"low": 0, "high": 1},
    "Beta": {"alpha": 2, "beta": 5},
    "Gamma": {"shape": 2, "scale": 2},
    "Inverse_Gamma": {"shape": 3, "scale": 2},  # Inverse computed later
    "Lognormal": {"mean": 0, "sigma": 1},
}

# Create the 'data' folder if it does not exist
data_folder = Path(__file__).parent / "data"
data_folder.mkdir(exist_ok=True)

def generate_distribution(name, params, size):
    """
    Generate data for a given distribution.

    Parameters:
        name (str): Name of the distribution.
        params (dict): Parameters for the distribution.
        size (int): Number of data points to generate.

    Returns:
        np.ndarray: Generated data.

    Raises:
        ValueError: If the distribution name is not supported.
    """
    if name == "Gaussian":
        return np.random.normal(loc=params["loc"], scale=params["scale"], size=size)
    elif name == "Uniform":
        return np.random.uniform(low=params["low"], high=params["high"], size=size)
    elif name == "Beta":
        return np.random.beta(a=params["alpha"], b=params["beta"], size=size)
    elif name == "Gamma":
        return np.random.gamma(shape=params["shape"], scale=params["scale"], size=size)
    elif name == "Inverse_Gamma":
        data = np.random.gamma(shape=params["shape"], scale=params["scale"], size=size)
        return 1 / data  # Compute inverse
    elif name == "Lognormal":
        return np.random.lognormal(mean=params["mean"], sigma=params["sigma"], size=size)
    else:
        raise ValueError(f"Unsupported distribution: {name}")

def save_distribution(data, file_name):
    """
    Save generated data to a CSV file.

    Parameters:
        data (np.ndarray): Data to save.
        file_name (str): Name of the output file.

    Returns:
        None
    """
    df = pd.DataFrame({"Value": data})
    output_file = data_folder / file_name
    df.to_csv(output_file, index=False)
    print(f"Saved distribution to {output_file}")

def main():
    """
    Main function to generate and save distributions.

    Iterates through predefined distributions, generates data, and saves 
    each distribution as a CSV file in the 'data' directory.

    Returns:
        None
    """
    for name, params in DISTRIBUTION_PARAMS.items():
        try:
            data = generate_distribution(name, params, NUM_POINTS)
            save_distribution(data, f"{name.lower()}_data.csv")
        except Exception as e:
            print(f"Error generating {name} distribution: {e}")

if __name__ == "__main__":
    """
    Entry point of the script. Calls the main function to generate and save
    predefined distributions.
    """
    main()

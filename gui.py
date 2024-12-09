import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import scipy.stats as stats
import io
import seaborn as sns
import streamlit as st

# Title of the app
st.title("CSV File Plotter and Data Distribution Explorer")

# Tabbed interface
tab1, tab2 = st.tabs(["CSV Plotter", "Distribution Plotter"])

# Function to plot the estimated density curve with optional overlay
def plot_density_with_overlay(data, overlay_distribution=None):
    """Plots the estimated density curve using KDE and optionally overlays a theoretical distribution curve."""
    fig, ax = plt.subplots()
    
    # Plot the KDE (kernel density estimate)
    sns.kdeplot(data, shade=True, color='blue', ax=ax, label='KDE Estimate')
    
    # Optionally overlay a theoretical distribution
    if overlay_distribution == "Normal":
        mean = np.mean(data)
        std_dev = np.std(data)
        x = np.linspace(min(data), max(data), 1000)
        ax.plot(x, stats.norm.pdf(x, loc=mean, scale=std_dev), color='red', label='Normal Distribution')
    elif overlay_distribution == "Uniform":
        lower = np.min(data)
        upper = np.max(data)
        x = np.linspace(lower, upper, 1000)
        ax.plot(x, stats.uniform.pdf(x, loc=lower, scale=upper-lower), color='red', label='Uniform Distribution')
    elif overlay_distribution == "Gamma":
        shape = 2.0  # Default shape parameter for Gamma distribution
        scale = 2.0  # Default scale parameter for Gamma distribution
        x = np.linspace(min(data), max(data), 1000)
        # Shift the Gamma distribution to center it around the data (moderate adjustment)
        loc = np.mean(data) - scale  # Reduced adjustment for a more subtle shift
        ax.plot(x, stats.gamma.pdf(x, shape, loc=loc, scale=scale), color='red', label='Gamma Distribution')
    elif overlay_distribution == "Inverse Gamma":
        shape = 3.0  # Default shape parameter for Inverse Gamma distribution
        scale = 2.0  # Default scale parameter for Inverse Gamma distribution
        x = np.linspace(min(data), max(data), 1000)
        # Align the maximum of both the data and the theoretical distribution
        mode = (shape - 1) * scale  # Mode of the Inverse Gamma distribution
        loc = np.mean(data) - mode  # Shift to align the maximum
        ax.plot(x, stats.invgamma.pdf(x, shape, loc=loc, scale=scale), color='red', label='Inverse Gamma Distribution')
    elif overlay_distribution == "Lognormal":
        mean = np.mean(data)
        std_dev = np.std(data)
        x = np.linspace(min(data), max(data), 1000)
        # Use `scale` for the mean and `shape` for the sigma (standard deviation of underlying normal distribution)
        ax.plot(x, stats.lognorm.pdf(x, std_dev, scale=mean), color='red', label='Lognormal Distribution')
    
    # Add labels and title
    ax.set_title("Density Curve with Optional Overlay")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    
    # Show legend
    ax.legend()
    
    # Display the plot
    st.pyplot(fig)
    plt.close(fig)  # Close the figure to avoid duplicate plots

# Tab 1: CSV Plotter
with tab1:
    st.header("CSV File Plotter")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load CSV data
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(data)

            # Check if the CSV has a 'Value' column
            if 'Value' not in data.columns:
                st.error("CSV file must contain a 'Value' column.")
            else:
                # Extract the 'Value' column data
                value_data = data['Value']

                # Option for overlaying a theoretical distribution
                overlay = st.selectbox(
                    "Select a theoretical distribution to overlay on the density plot",
                    ["None", "Normal", "Uniform", "Gamma", "Inverse Gamma", "Lognormal"],
                    help="Choose a distribution to compare against the estimated density curve."
                )

                # Save the plot to a buffer
                buf = io.BytesIO()
                plot_density_with_overlay(value_data, overlay)  # This will save the plot into buf
                buf.seek(0)

                # Download button for the plot
                st.download_button(
                    label="Download Plot",
                    data=buf,
                    file_name="density_plot.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a CSV file to get started.")

# Tab 2: Distribution Plotter
with tab2:
    st.header("Data Distribution Plotter")

    # Dropdown to select distribution
    distribution = st.selectbox(
        "Select a data distribution",
        ["Gaussian (Normal)", "Uniform", "Exponential", "Gamma", "Inverse Gamma", "Lognormal"],
        help="Choose a theoretical distribution to visualize."
    )

    # Number of samples
    num_samples = st.slider("Number of samples", min_value=100, max_value=5000, value=1000, step=100, help="Number of data points to generate.")

    # Generate data
    data = None
    if distribution == "Gaussian (Normal)":
        mean = st.number_input("Mean", value=0.0, step=0.1, help="Mean of the normal distribution.")
        std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, help="Standard deviation of the normal distribution.")
        data = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
    elif distribution == "Uniform":
        lower = st.number_input("Lower Bound", value=0.0, step=0.1, help="Minimum value of the uniform distribution.")
        upper = st.number_input("Upper Bound", value=1.0, step=0.1, help="Maximum value of the uniform distribution.")
        if upper > lower:
            data = np.random.uniform(low=lower, high=upper, size=num_samples)
        else:
            st.error("Upper bound must be greater than lower bound.")
    elif distribution == "Gamma":
        shape = st.number_input("Shape (k)", value=2.0, step=0.1, help="Shape parameter of the gamma distribution.")
        scale = st.number_input("Scale (theta)", value=2.0, step=0.1, help="Scale parameter of the gamma distribution.")
        data = np.random.gamma(shape=shape, scale=scale, size=num_samples)
    elif distribution == "Inverse Gamma":
        shape = st.number_input("Shape (alpha)", value=3.0, step=0.1, help="Shape parameter of the inverse gamma distribution.")
        scale = st.number_input("Scale (beta)", value=2.0, step=0.1, help="Scale parameter of the inverse gamma distribution.")
        if shape > 0 and scale > 0:
            data = stats.invgamma.rvs(a=shape, scale=scale, size=num_samples)
        else:
            st.error("Shape and scale parameters must be positive.")
    elif distribution == "Lognormal":
        mean = st.number_input("Mean", value=0.0, step=0.1, help="Mean of the Lognormal distribution.")
        std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, help="Standard deviation of the Lognormal distribution.")
        data = np.random.lognormal(mean=mean, sigma=std_dev, size=num_samples)

    # Plot the generated data
    if data is not None:
        fig, ax = plt.subplots()
        bins = st.slider("Histogram Bin Size", min_value=10, max_value=100, value=30, step=5, key='dist_bin_size')
        ax.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(f"{distribution} Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        
        # Overlay the theoretical distribution density
        if distribution == "Gaussian (Normal)":
            mean = np.mean(data)
            std_dev = np.std(data)
            x = np.linspace(min(data), max(data), 1000)
            ax.plot(x, stats.norm.pdf(x, loc=mean, scale=std_dev), color='red', label='Normal Distribution')
        elif distribution == "Uniform":
            lower = np.min(data)
            upper = np.max(data)
            x = np.linspace(lower, upper, 1000)
            ax.plot(x, stats.uniform.pdf(x, loc=lower, scale=upper-lower), color='red', label='Uniform Distribution')
        elif distribution == "Gamma":
            shape = 2.0  # Default shape parameter for Gamma distribution
            scale = 2.0  # Default scale parameter for Gamma distribution
            x = np.linspace(min(data), max(data), 1000)
            # Shift the Gamma distribution to center it around the data (moderate adjustment)
            loc = np.mean(data) - scale  # Reduced adjustment for a more subtle shift
            ax.plot(x, stats.gamma.pdf(x, shape, loc=loc, scale=scale), color='red', label='Gamma Distribution')
        elif distribution == "Inverse Gamma":
            shape = 3.0  # Default shape parameter for Inverse Gamma distribution
            scale = 2.0  # Default scale parameter for Inverse Gamma distribution
            x = np.linspace(min(data), max(data), 1000)
            # Align the maximum of both the data and the theoretical distribution
            mode = (shape - 1) * scale  # Mode of the Inverse Gamma distribution
            loc = np.mean(data) - mode  # Shift to align the maximum
            ax.plot(x, stats.invgamma.pdf(x, shape, loc=loc, scale=scale), color='red', label='Inverse Gamma Distribution')
        elif distribution == "Lognormal":
            mean = np.mean(data)
            std_dev = np.std(data)
            x = np.linspace(min(data), max(data), 1000)
            ax.plot(x, stats.lognorm.pdf(x, std_dev, scale=mean), color='red', label='Lognormal Distribution')
        
        # Show the legend and plot
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        # Save the plot to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png")  # Save the plot correctly into buf
        buf.seek(0)

        # Download button for the plot
        st.download_button(
            label="Download Plot",
            data=buf,
            file_name="distribution_plot.png",
            mime="image/png"
        )

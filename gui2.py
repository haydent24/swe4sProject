import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Density Curve Plotter and Data Distribution Explorer")

# Tabbed interface
tab1, tab2 = st.tabs(["CSV Density Curve", "Sample Distributions"])

# Tab 1: CSV Density Curve
with tab1:
    st.header("Density Curve Plotter")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file (single column)", type="csv")

    if uploaded_file is not None:  # Ensure a file has been uploaded
        try:
            # Load CSV data
            data = pd.read_csv(uploaded_file)
            
            # Check if the file has exactly one column
            if data.shape[1] == 1:
                # Extract the single column
                column_name = data.columns[0]
                st.write(f"Loaded column: **{column_name}**")
                st.dataframe(data)

                # Dropdown to customize the density plot
                kde_bw = st.slider(
                    "Bandwidth for KDE (higher values smooth the curve)", 
                    min_value=0.1, 
                    max_value=2.0, 
                    value=1.0, 
                    step=0.1
                )
                bins = st.slider(
                    "Number of histogram bins (for visualization)", 
                    min_value=10, 
                    max_value=100, 
                    value=30, 
                    step=5
                )

                # Plot density curve with histogram
                fig, ax = plt.subplots()
                sns.histplot(data[column_name], kde=True, bins=bins, kde_kws={'bw_adjust': kde_bw}, ax=ax)
                ax.set_title(f"Density Curve for {column_name}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                st.pyplot(fig)
            else:
                st.error("The uploaded CSV file must have exactly one column.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a CSV file with a single column to plot its density curve.")

# Tab 2: Sample Distributions
with tab2:
    st.header("Sample Distributions Plotter")

    # Dropdown to select distribution
    distribution = st.selectbox(
        "Select a data distribution",
        ["Gaussian (Normal)", "Uniform", "Exponential", "Gamma", "Inverse Gamma"]
    )

    # Number of samples
    num_samples = st.slider("Number of samples", min_value=100, max_value=5000, value=1000, step=100)

    # Generate data based on the selected distribution
    data = None  # Initialize data variable
    if distribution == "Gaussian (Normal)":
        mean = st.number_input("Mean", value=0.0, step=0.1)
        std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1)
        data = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
        st.write(f"Generated {num_samples} samples from a Gaussian distribution (mean={mean}, std={std_dev}).")
    elif distribution == "Uniform":
        lower = st.number_input("Lower Bound", value=0.0, step=0.1)
        upper = st.number_input("Upper Bound", value=1.0, step=0.1)
        if upper <= lower:
            st.error("Upper bound must be greater than lower bound.")
        else:
            data = np.random.uniform(low=lower, high=upper, size=num_samples)
            st.write(f"Generated {num_samples} samples from a Uniform distribution (range={lower}-{upper}).")
    elif distribution == "Exponential":
        scale = st.number_input("Scale (1/lambda)", value=1.0, step=0.1)
        data = np.random.exponential(scale=scale, size=num_samples)
        st.write(f"Generated {num_samples} samples from an Exponential distribution (scale={scale}).")
    elif distribution == "Gamma":
        shape = st.number_input("Shape (k)", value=2.0, step=0.1)
        scale = st.number_input("Scale (theta)", value=2.0, step=0.1)
        data = np.random.gamma(shape=shape, scale=scale, size=num_samples)
        st.write(f"Generated {num_samples} samples from a Gamma distribution (shape={shape}, scale={scale}).")
    elif distribution == "Inverse Gamma":
        shape = st.number_input("Shape (alpha)", value=3.0, step=0.1)
        scale = st.number_input("Scale (beta)", value=2.0, step=0.1)
        if shape <= 0 or scale <= 0:
            st.error("Shape and scale parameters must be positive.")
        else:
            data = 1 / np.random.gamma(shape=shape, scale=1/scale, size=num_samples)
            st.write(f"Generated {num_samples} samples from an Inverse Gamma distribution (shape={shape}, scale={scale}).")

    # Plot the generated data
    if data is not None:
        fig, ax = plt.subplots()
        ax.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(f"{distribution} Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

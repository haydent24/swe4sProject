import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title("CSV File Plotter and Data Distribution Explorer")

# Tabbed interface
tab1, tab2 = st.tabs(["CSV Plotter", "Distribution Plotter"])

# Tab 1: CSV Plotter
with tab1:
    st.header("CSV File Plotter")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:  # Ensure a file has been uploaded
        try:
            # Load CSV data
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(data)

            # Check if the file has at least two columns
            if len(data.columns) >= 2:
                # Dropdowns to select columns for X and Y axes
                x_col = st.selectbox("Select X-axis column", data.columns, key="x_col")
                y_col = st.selectbox("Select Y-axis column", data.columns, key="y_col")

                # Dropdown to select the graph type
                graph_type = st.selectbox(
                    "Select graph type",
                    ["Line", "Scatter", "Bar"],
                    key="graph_type"
                )

                # Plotting based on selected graph type
                fig, ax = plt.subplots()

                if graph_type == "Line":
                    ax.plot(data[x_col], data[y_col], marker='o', label=f"{y_col} vs {x_col}")
                elif graph_type == "Scatter":
                    ax.scatter(data[x_col], data[y_col], label=f"{y_col} vs {x_col}")
                elif graph_type == "Bar":
                    ax.bar(data[x_col], data[y_col], label=f"{y_col} vs {x_col}")

                # Customize the plot
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{graph_type} Plot")
                ax.legend()
                ax.grid(True)

                # Show the plot
                st.pyplot(fig)
            else:
                st.error("The CSV file must have at least two columns.")
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

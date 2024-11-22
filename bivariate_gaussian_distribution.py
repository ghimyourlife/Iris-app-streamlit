import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from scipy.stats import multivariate_normal  # Import multivariate normal distribution from SciPy
import streamlit as st  # Import Streamlit for building the web app

# Configure Matplotlib parameters
p = plt.rcParams  # Access Matplotlib runtime configuration
p["font.sans-serif"] = ["DejaVu Sans"]  # Set the default font to DejaVu Sans
p["font.weight"] = "light"  # Set font weight to light
p["ytick.minor.visible"] = True  # Enable visibility of minor ticks on the y-axis
p["xtick.minor.visible"] = True  # Enable visibility of minor ticks on the x-axis
p["axes.grid"] = True  # Enable grid for the plots
p["grid.color"] = "0.5"  # Set grid color to a medium gray
p["grid.linewidth"] = 0.5  # Set grid line width

# Sidebar inputs for user interaction
with st.sidebar:
    st.title('Bivariate Gaussian Distribution')  # Add a title to the sidebar

    # Mean for the first dimension
    mu_X1 = st.slider('mu_X1', min_value=-4.0, max_value=4.0, value=0.0, step=0.1)  # Slider for mu_X1
    # Mean for the second dimension
    mu_X2 = st.slider('mu_X2', min_value=-4.0, max_value=4.0, value=0.0, step=0.1)  # Slider for mu_X2
    # Standard deviation for the first dimension
    sigma_X1 = st.slider('sigma_X1', min_value=0.5, max_value=3.0, value=1.0, step=0.1)  # Slider for sigma_X1
    # Standard deviation for the second dimension
    sigma_X2 = st.slider('sigma_X2', min_value=0.5, max_value=3.0, value=1.0, step=0.1)  # Slider for sigma_X2
    # Correlation coefficient
    rho = st.slider('rho', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)  # Slider for rho

# Set the mean vector
mu = [mu_X1, mu_X2]  # Mean vector for the bivariate distribution

# Construct the covariance matrix
Sigma = [[sigma_X1**2, sigma_X1 * sigma_X2 * rho],  # Covariance matrix element (1, 2)
         [sigma_X1 * sigma_X2 * rho, sigma_X2**2]]  # Covariance matrix element (2, 2)

# Create a grid of values for x1 and x2
width = 4  # Range of the grid
x1 = np.linspace(-width, width, 321)  # Linearly spaced values for x1
x2 = np.linspace(-width, width, 321)  # Linearly spaced values for x2

# Generate a meshgrid for x1 and x2
xx1, xx2 = np.meshgrid(x1, x2)  # Create grid points for x1 and x2

# Combine x1 and x2 values into pairs for PDF calculation
xx12 = np.dstack((xx1, xx2))  # Stack grid points into pairs

# Create a bivariate normal distribution
bi_norm = multivariate_normal(mu, Sigma)  # Define the bivariate normal distribution

# Compute the PDF for the grid
PDF_joint = bi_norm.pdf(xx12)  # Calculate the joint PDF for the grid

# Create a plot for the bivariate Gaussian PDF
fig, ax = plt.subplots(figsize=(5, 5))  # Initialize a figure with specified size

# Plot the contour map of the PDF
plt.contourf(xx1, xx2, PDF_joint, 20, cmap='RdYlBu_r')  # Filled contours with 20 levels and color map

# Add vertical and horizontal lines indicating the means
plt.axvline(x=mu_X1, color='k', linestyle='--')  # Vertical line at mu_X1
plt.axhline(y=mu_X2, color='k', linestyle='--')  # Horizontal line at mu_X2

# Set axis labels
ax.set_xlabel('$x_1$')  # Label for x-axis
ax.set_ylabel('$x_2$')  # Label for y-axis

# Display the plot in the Streamlit app
st.pyplot(fig)  # Render the plot using Streamlit
#st.markdown("Code download please visit [Github Repo: Visualize-ML](https://github.com/visualize-ml)")
st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)

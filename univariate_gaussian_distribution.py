# import modules
import streamlit as st  
from scipy.stats import beta  
import matplotlib.pyplot as plt  
import numpy as np  

# Configure Matplotlib parameters
p = plt.rcParams  
p["font.sans-serif"] = ["DejaVu Sans"]  # Set the default font to DejaVu Sans
p["font.weight"] = "light"  # Set font weight to light
p["ytick.minor.visible"] = True  # Enable visibility of minor ticks on the y-axis
p["xtick.minor.visible"] = True  # Enable visibility of minor ticks on the x-axis
p["axes.grid"] = True  # Enable grid for the plots
p["grid.color"] = "0.5"  # Setting grid color to a medium gray
p["grid.linewidth"] = 0.5  # Setting grid line width

# Define a function for the univariate Gaussian PDF
def uni_normal_pdf(x, mu, sigma):
    coeff = 1 / np.sqrt(2 * np.pi) / sigma  # Coefficient of the PDF formula
    z = (x - mu) / sigma  # Standardized variable (z-score)
    f_x = coeff * np.exp(-1 / 2 * z**2)  # Calculating the PDF value
    return f_x  # Returning the PDF value

x_array = np.linspace(-5, 5, 200)  # Linearly spaced values between -5 and 5

# Create a sidebar for user inputs
with st.sidebar:
    st.title('Univariate Gaussian distribution PDF')  # Adding a title in the sidebar
    st.latex(r'''{\displaystyle f(x)={\frac {1}{\sigma {\sqrt {2\pi }}}}
             e^{-{\frac {1}{2}}\left({
             \frac {x-\mu }{\sigma }}\right)^{2}}}''')  # Displaying the Gaussian PDF formula in LaTeX
    mu_input = st.slider('mu', min_value=-5.0, max_value=5.0, value=0.0, step=0.2)  # Slider for mean (mu)
    sigma_input = st.slider('sigma', min_value=0.0, max_value=4.0, value=1.0, step=0.1)  # Slider for standard deviation (sigma)

pdf_array = uni_normal_pdf(x_array, mu_input, sigma_input)  # Calculate PDF values for the user-defined mu and sigma

fig, ax = plt.subplots(figsize=(8, 5))  # Initialize a figure with specified dimensions

ax.plot(x_array, pdf_array, 'b', lw=1)  # Plot the PDF as a blue line

ax.axvline(x=mu_input, c='r', ls='--')  # Vertical line at the mean
ax.axvline(x=mu_input + sigma_input, c='r', ls='--')  # Vertical line at mu + sigma
ax.axvline(x=mu_input - sigma_input, c='r', ls='--')  # Vertical line at mu - sigma

ax.plot(x_array, uni_normal_pdf(x_array, 0, 1), c=[0.8, 0.8, 0.8], lw=1)  # Plot standard normal distribution for comparison
# Add vertical lines at 0, ±1 for the standard normal distribution
ax.axvline(x=0, c=[0.8, 0.8, 0.8], ls='--')  # Vertical line at mean 0
ax.axvline(x=0 + 1, c=[0.8, 0.8, 0.8], ls='--')  # Vertical line at mean +1
ax.axvline(x=0 - 1, c=[0.8, 0.8, 0.8], ls='--')  # Vertical line at mean -1

ax.set_xlim(-5, 5)  # Set x-axis limits
ax.set_ylim(0, 1)  # Set y-axis limits
ax.set_xlabel(r'$x$')  # Label for the x-axis
ax.set_ylabel(r'$f_X(x)$')  # Label for the y-axis
ax.spines.right.set_visible(False)  # Hide the right spine
ax.spines.top.set_visible(False)  # Hide the top spine
ax.yaxis.set_ticks_position('left')  # Position ticks on the left for y-axis
ax.xaxis.set_ticks_position('bottom')  # Position ticks on the bottom for x-axis
ax.tick_params(axis="x", direction='in')  # Set x-axis tick direction inward
ax.tick_params(axis="y", direction='in')  # Set y-axis tick direction inward

st.pyplot(fig)  # Render the plot using Streamlit
st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)

# import modules
import plotly.graph_objects as go  
import numpy as np  
import streamlit as st 
from scipy.stats import multivariate_normal  

# Display the formula for multivariate Gaussian distribution in LaTeX
st.latex(r'''{\displaystyle f_{\mathbf {X} }(x_{1},\ldots ,x_{k})=
         {\frac {\exp \left(-{\frac {1}{2}}
         ({\mathbf {x} }-{\boldsymbol {\mu }})
         ^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}
         ({\mathbf {x} }-{\boldsymbol {\mu }})\right)}
         {\sqrt {(2\pi )^{k}|{\boldsymbol {\Sigma }}|}}}}''')

# Define a function to create a LaTeX bmatrix for the covariance matrix
def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:  # Ensure the array has at most 2 dimensions
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # Prepare the array lines for LaTeX formatting
    rv = [r'\begin{bmatrix}']  # Start the LaTeX bmatrix
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # Add each line formatted with LaTeX
    rv +=  [r'\end{bmatrix}']  # Close the bmatrix
    return '\n'.join(rv)  # Join the lines with newlines

xxx1,xxx2,xxx3 = np.mgrid[-3:3:0.2,-3:3:0.2,-3:3:0.2]  # Create a 3D grid with specified intervals

with st.sidebar:  # Create a sidebar for user inputs
    st.title('Trivariate Gaussian Distribution')  # Add a title to the sidebar
    
    # Sliders for standard deviations
    sigma_1 = st.slider('sigma_1', min_value=0.5, max_value=3.0, value=1.0, step=0.1)  # Standard deviation for dimension 1
    sigma_2 = st.slider('sigma_2', min_value=0.5, max_value=3.0, value=1.0, step=0.1)  # Standard deviation for dimension 2
    sigma_3 = st.slider('sigma_3', min_value=0.5, max_value=3.0, value=1.0, step=0.1)  # Standard deviation for dimension 3
    
    # Sliders for correlation coefficients
    rho_1_2 = st.slider('rho_1_2', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)  # Correlation between dimension 1 and 2
    rho_1_3 = st.slider('rho_1_3', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)  # Correlation between dimension 1 and 3
    rho_2_3 = st.slider('rho_2_3', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)  # Correlation between dimension 2 and 3
    
# Construct the covariance matrix
SIGMA = np.array([[sigma_1**2, rho_1_2*sigma_1*sigma_2, rho_1_3*sigma_1*sigma_3],  # First row of the covariance matrix
                  [rho_1_2*sigma_1*sigma_2, sigma_2**2, rho_2_3*sigma_2*sigma_3],  # Second row
                  [rho_1_3*sigma_1*sigma_3, rho_2_3*sigma_2*sigma_3, sigma_3**2]])  # Third row

st.latex(r'\Sigma = ' + bmatrix(SIGMA))  # Convert the covariance matrix to LaTeX format

MU = np.array([0, 0, 0])  # Mean vector is set to zero for all dimensions

pos = np.dstack((xxx1.ravel(),xxx2.ravel(),xxx3.ravel()))  # Combine the grid points into coordinate pairs

rv  = multivariate_normal(MU, SIGMA)  # Create the multivariate Gaussian distribution

PDF = rv.pdf(pos)  # Evaluate the PDF for the grid points

# Create a 3D visualization of the PDF using Plotly
fig = go.Figure(data=go.Volume(
    x=xxx1.flatten(),  # Flattened x-coordinates
    y=xxx2.flatten(),  # Flattened y-coordinates
    z=xxx3.flatten(),  # Flattened z-coordinates
    value=PDF.flatten(),  # Flattened PDF values
    isomin=0,  # Minimum iso-surface value
    isomax=PDF.max(),  # Maximum iso-surface value
    colorscale='RdYlBu_r',  # Reversed red-yellow-blue color scale
    opacity=0.1,  # Set transparency
    surface_count=11,  # Number of iso-surfaces
    ))

# Customize the layout of the 3D plot
fig.update_layout(scene=dict(
                    xaxis_title=r'x_1',  # Label for x-axis
                    yaxis_title=r'x_2',  # Label for y-axis
                    zaxis_title=r'x_3'),  # Label for z-axis
                    width=1000,  # Set the figure width
                    margin=dict(r=20, b=10, l=10, t=10))  # Set margins for the plot

st.plotly_chart(fig, theme=None, use_container_width=True)  # Render the plot using Streamlit
st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)
import streamlit as st  # Import Streamlit for interactive UI
import matplotlib.pyplot as plt  # Import Matplotlib for basic plotting
import numpy as np  # Import NumPy for numerical computations
import plotly.graph_objects as go  # Import Plotly for 3D visualization

with st.sidebar:  # Create a sidebar for user interaction
    st.title('3D Lp Norm')  # Set sidebar title
    p = st.slider('p', 1.0, 20.0, 1.0, 0.1)  # Create a slider to select the p-value for Lp norm

x1 = np.linspace(-2.5, 2.5, num=21)  # Generate 21 evenly spaced values in the range [-2.5, 2.5] for x1
x2 = x1  # Use the same values for x2
x3 = x1  # Use the same values for x3

xxx1, xxx2, xxx3 = np.meshgrid(x1, x2, x3)  # Create a 3D coordinate grid for (x1, x2, x3)

# Define a function to compute the Lp norm in 3D space
def Lp_norm(p):  # Function to compute Lp norm
    if np.isinf(p):  # Check if p is infinity
        zz = np.maximum(np.abs(xxx1), np.abs(xxx2), np.abs(xxx3))  # Compute Chebyshev norm (max absolute value of coordinates)
    else:
        zz = ((np.abs(xxx1)**p) +  # Compute the Lp norm formula
              (np.abs(xxx2)**p) +  
              (np.abs(xxx3)**p))**(1./p)  
    return zz  # Return the computed norm values

zzz = Lp_norm(p)  # Compute the Lp norm using the selected p value

fig = go.Figure(data=go.Volume(  # Create a 3D volume plot using Plotly
    x=xxx1.flatten(),  # Flatten the 3D x-coordinates for plotting
    y=xxx2.flatten(),  # Flatten the 3D y-coordinates for plotting
    z=xxx3.flatten(),  # Flatten the 3D z-coordinates for plotting
    value=zzz.flatten(),  # Flatten the computed Lp norm values
    opacity=0.18,  # Set transparency for better visualization
    surface_count=10,  # Number of contour surfaces to be displayed
    colorscale='RdYlBu'  # Use a red-yellow-blue color scale for better contrast
))

fig.update_layout(autosize=False,  # Disable automatic figure resizing
                  width=800, height=800,  # Set figure width and height
                  margin=dict(l=65, r=50, b=65, t=90))  # Adjust figure margins

st.plotly_chart(fig)  # Display the 3D Plotly figure in the Streamlit app



st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)

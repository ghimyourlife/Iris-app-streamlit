import numpy as np  # Import NumPy for numerical computations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import matplotlib as mpl  # Import additional Matplotlib settings
from matplotlib import cm  # Import color mapping module
import plotly.graph_objects as go  # Import Plotly for creating interactive visualizations
import streamlit as st  # Import Streamlit for creating the web app

# Configure Matplotlib parameters
p = plt.rcParams  # Get the global Matplotlib settings
p["font.sans-serif"] = ["DejaVu Sans"]  # Set the font to 'DejaVu Sans'
p["font.weight"] = "light"  # Set the font weight to 'light'
p["ytick.minor.visible"] = False  # Disable minor ticks on the y-axis
p["xtick.minor.visible"] = False  # Disable minor ticks on the x-axis
p["axes.grid"] = True  # Enable grid lines
p["grid.color"] = "0.5"  # Set grid line color to gray
p["grid.linewidth"] = 0.5  # Set grid line thickness

def bmatrix(a):  # Define a function to generate a LaTeX bmatrix
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:  # Ensure the input array has at most 2 dimensions
        raise ValueError('bmatrix can at most display two dimensions')  # Raise an error for invalid input
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # Format the array into strings
    rv = [r'\begin{bmatrix}']  # Start the LaTeX bmatrix
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # Format each row
    rv += [r'\end{bmatrix}']  # End the LaTeX bmatrix
    return '\n'.join(rv)  # Combine into a single LaTeX string

with st.sidebar:  # Streamlit sidebar configuration
    st.title('Volume Plot')  # Add a title in the sidebar
    st.latex(r'''Q = \begin{bmatrix}
    a & b & c\\
    b & d & e\\
    c & e & f
    \end{bmatrix}''' )  # Display the matrix formula in LaTeX format
    
    a = st.slider('a', -5, 5, 1, 1)  # Slider for 'a' with range -5 to 5 and step size of 1
    b = st.slider('b', -5, 5, 0, 1)  # Slider for 'b'
    c = st.slider('c', -5, 5, 0, 1)  # Slider for 'c'
    d = st.slider('d', -5, 5, 2, 1)  # Slider for 'd'
    e = st.slider('e', -5, 5, 0, 1)  # Slider for 'e'
    f = st.slider('f', -5, 5, 3, 1)  # Slider for 'f'

x = np.linspace(-2, 2, 21)  # Generate 21 points between -2 and 2 for the x-axis
y = np.linspace(-2, 2, 21)  # Generate 21 points for the y-axis
z = np.linspace(-2, 2, 21)  # Generate 21 points for the z-axis

X, Y, Z = np.meshgrid(x, y, z)  # Create a 3D grid of x, y, z coordinates

Points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # Flatten and combine x, y, z into a 2D array

Q = np.array([[a, b, c],  # Define the symmetric matrix Q using the slider values
              [b, d, e],
              [c, e, f]])

st.latex('Q =' + bmatrix(Q))  # Show the matrix Q in the sidebar

fff = np.diag(Points @ Q @ Points.T)  # Compute the quadratic form for all points

fig = go.Figure(data=go.Volume(  # Create a 3D volume plot
    x=X.flatten(),  # Flatten the x-coordinates
    y=Y.flatten(),  # Flatten the y-coordinates
    z=Z.flatten(),  # Flatten the z-coordinates
    value=fff.flatten(),  # Flatten the scalar values
    opacity=0.18,  # Set the opacity for the volume rendering
    surface_count=11,  # Define the number of isosurfaces
    colorscale='RdYlBu'  # Use the red-yellow-blue colormap
    ))
fig.update_layout(autosize=False,  # Disable automatic resizing
                  width=800, height=800,  # Set the figure width and height
                  margin=dict(l=65, r=50, b=65, t=90))  # Define margins for the layout

st.plotly_chart(fig)  # Embed the 3D volume plot in the Streamlit app


st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)
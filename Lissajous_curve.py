import time  # Import the time module for potential timing operations
import numpy as np  # Import numpy for numerical computations
import streamlit as st  # Import streamlit for web-based UI interactions
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from matplotlib.collections import LineCollection  # Import LineCollection for handling line segments in the plot
from math import gcd  # Import gcd (Greatest Common Divisor) function for calculating the fundamental period

with st.sidebar:  # Create a sidebar for user input
    st.title('Lissajous curves')  # Display the title in the sidebar
    nx = st.slider('nx', 1, 10, 1, 1)  # Slider for selecting nx values from 1 to 10
    ny = st.slider('ny', 1, 10, 1, 1)  # Slider for selecting ny values from 1 to 10
    k  = st.slider('k', 1, 10, 1, 1)  # Slider for selecting k values from 1 to 10

t = np.linspace(0, 1/gcd(nx, ny), 1000)  # Generate 1000 time values over the fundamental period defined by gcd(nx, ny)

x_traj = np.cos(2*np.pi*nx*t)  # Compute x-coordinates of the Lissajous curve
y_traj = np.cos(2*np.pi*ny*t + k*np.pi/4/nx)  # Compute y-coordinates with phase shift based on k and nx

points = np.array([x_traj, y_traj]).T.reshape(-1, 1, 2)  # Reshape trajectory points to fit for segment creation
segments = np.concatenate([points[:-1], points[1:]], axis=1)  # Create line segments by connecting consecutive points

norm = plt.Normalize(t.min(), t.max())  # Normalize time values for color mapping
    
lc = LineCollection(segments, cmap='hsv', norm=norm)  # Create a LineCollection with HSV colormap to visualize trajectory
lc.set_array(t)  # Set the color mapping array based on the time values
lc.set_linewidth(2)  # Set the width of the plotted lines

fig, ax = plt.subplots(figsize=(3,3))  # Create a figure and axis with a size of 3x3 inches
ax.add_collection(lc)  # Add the colored line segments to the plot
plt.axis('equal')  # Ensure equal scaling for both axes
plt.axis('off')  # Remove axis ticks and labels for a cleaner visualization
plt.show()  # Display the plot
st.pyplot(fig)  # Render the figure in the Streamlit app

st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)

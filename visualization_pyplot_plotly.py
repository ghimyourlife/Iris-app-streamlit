# import modules
import plotly.graph_objects as go  
import numpy as np  
import matplotlib.pyplot as plt  
import streamlit as st  

x1_array = np.linspace(-3, 3, 301)  # Create x1 values from -3 to 3
x2_array = np.linspace(-3, 3, 301)  # Create x2 values from -3 to 3
xx1, xx2 = np.meshgrid(x1_array, x2_array)  # Create a grid for 3D plotting

ff = xx1 * np.exp(-xx1**2 - xx2**2)  # Calculate surface values for the function

fig = plt.figure(figsize=(8, 8))  # Create a figure with specified size
ax = fig.add_subplot(projection='3d')  # Add a 3D subplot
ax.plot_wireframe(xx1, xx2, ff, rstride=10, cstride=10)  # Plot a 3D wireframe
st.pyplot(fig)  # a. Display the static plot in Streamlit

fig = go.Figure(data=[go.Surface(z=ff, x=xx1, y=xx2,
                      colorscale='RdYlBu_r')]) # Create a Plotly figure

st.plotly_chart(fig)  # b. Display the interactive plot in Streamlit
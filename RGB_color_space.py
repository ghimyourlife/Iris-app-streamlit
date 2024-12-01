import plotly.graph_objects as go
import numpy as np
import streamlit as st

# Create a sidebar in Streamlit
with st.sidebar:
    st.title('RGB Color Space')  # Sidebar title
    num = st.slider('Granularity', 5, 15, 8, 1)  # Slider to select the granularity

# Generate RGB color values with the specified granularity
Red   = np.linspace(0, 255, num)
Green = np.linspace(0, 255, num)
Blue  = np.linspace(0, 255, num)

# Create a 3D grid of RGB values
RRR, GGG, BBB = np.meshgrid(Red, Green, Blue)
colors_rgb = np.column_stack((RRR.ravel(),
                              GGG.ravel(),
                              BBB.ravel()))  # Combine RGB values into a single array

# Extract R, G, and B values
r_values, g_values, b_values = zip(*colors_rgb)

# Create a 3D scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=r_values,  # Red values on the x-axis
    y=g_values,  # Green values on the y-axis
    z=b_values,  # Blue values on the z-axis
    mode='markers',  # Scatter plot in marker mode
    marker=dict(
        color=colors_rgb,  # Use RGB colors for the markers
        size=6,  # Set the size of the markers
    )
))

# Configure the layout of the 3D plot
fig.update_layout(
    scene=dict(
        xaxis=dict(title='Red'),  # Label for the x-axis
        yaxis=dict(title='Green'),  # Label for the y-axis
        zaxis=dict(title='Blue')),  # Label for the z-axis
    margin=dict(l=0, r=0, b=0, t=0))  # Remove extra margins
fig.layout.scene.camera.projection.type = "orthographic"  # Use orthographic projection

# Display the plot in Streamlit
st.plotly_chart(fig)

st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)

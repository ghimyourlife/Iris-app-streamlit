import streamlit as st  # Import Streamlit for interactive UI
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import numpy as np  # Import NumPy for numerical operations

# Create a sidebar for user interaction
with st.sidebar:
    st.title('Lp Norm in 2D')  # Set sidebar title
    p = st.slider('p', 1.0, 20.0, 1.0, 0.1)  # Create a slider for selecting the value of p

x1 = np.linspace(-2.5, 2.5, num=101)  # Generate 101 evenly spaced values in the range [-2.5, 2.5]
x2 = x1  # Use the same values for x2

xx1, xx2 = np.meshgrid(x1, x2)  # Create a coordinate grid for x1 and x2

# Define a function to compute the Lp norm
def Lp_norm(p):  # Function to compute the Lp norm
    if np.isinf(p):  # Check if p is infinity
        zz = np.maximum(np.abs(xx1), np.abs(xx2))  # Compute Chebyshev norm (max absolute value)
    else:  
        zz = ((np.abs(xx1)**p) + (np.abs(xx2)**p))**(1./p)  # Compute the Lp norm using the given formula
    return zz  # Return the computed norm values

fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure and axis with a 6x6 aspect ratio

zz = Lp_norm(p)  # Compute the Lp norm using the selected p value

ax.contourf(xx1, xx2, zz, 20, cmap='RdYlBu_r')  # Plot filled contour using the reversed red-yellow-blue colormap

ax.contour(xx1, xx2, zz, [1], colors='k', linewidths=2)  # Draw a black contour line where Lp norm = 1

ax.axhline(y=0, color='k', linewidth=0.25)  # Add a thin black horizontal line at y=0
ax.axvline(x=0, color='k', linewidth=0.25)  # Add a thin black vertical line at x=0
ax.set_xlim(-2.5, 2.5)  # Set x-axis limits to [-2.5, 2.5]
ax.set_ylim(-2.5, 2.5)  # Set y-axis limits to [-2.5, 2.5]

ax.spines['top'].set_visible(False)  # Hide the top frame line
ax.spines['right'].set_visible(False)  # Hide the right frame line
ax.spines['bottom'].set_visible(False)  # Hide the bottom frame line
ax.spines['left'].set_visible(False)  # Hide the left frame line

ax.set_xticks([])  # Remove x-axis tick marks
ax.set_yticks([])  # Remove y-axis tick marks

ax.set_title('p = ' + str(p))  # Set title showing the current p value

ax.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio for better visualization

st.pyplot(fig)  # Display the figure in the Streamlit application


st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)

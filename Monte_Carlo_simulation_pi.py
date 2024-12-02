# import libraries
import streamlit as st  
import numpy as np  
import matplotlib.pyplot as plt  

# Configure Matplotlib plot settings
p = plt.rcParams  
p["font.sans-serif"] = ["DejaVu Sans"]  # Set the font to DejaVu Sans
p["font.weight"] = "light"  # Set the font weight to light
p["ytick.minor.visible"] = False  # Disable visibility of minor y-axis ticks
p["xtick.minor.visible"] = False  # Disable visibility of minor x-axis ticks
p["axes.grid"] = True  # Enable gridlines on the plot
p["grid.color"] = "0.5"  # Set the grid color to gray
p["grid.linewidth"] = 0.5  # Set the gridline width

# (1) Create a sidebar in Streamlit
with st.sidebar:  
    st.title('Estimate π')  # Add a title to the sidebar
    num = st.slider('Number of points', 200, 5000, 200, 200)  # Add a slider to select the number of scatter points

X = np.random.uniform(low=-1, high=1, size=(num, 2))  # Generate 2D points uniformly distributed in [-1, 1]
mask_inside = (X[:, 0]**2 + X[:, 1]**2 <= 1)  # Create a mask for points inside or on the unit circle
fig, ax = plt.subplots()  # Create a figure and axis object for plotting
X_inside = X[mask_inside, :]  # Points inside (or on) the unit circle
X_outside = X[~mask_inside, :]  # Points outside the unit circle
colors = np.array(['#377eb8', '#ff7f00'])  # Define colors: blue for inside, orange for outside
circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None')  # Create a circle with radius 1, black border, no fill
ax.add_patch(circ)  # Add the circle to the axis
plt.scatter(X_inside[:, 0], X_inside[:, 1], color=colors[0], marker='.')  # Plot inside points in blue
plt.scatter(X_outside[:, 0], X_outside[:, 1], color=colors[1], marker='x')  # Plot outside points in orange
ax.set_aspect('equal', adjustable='box')  # Set equal aspect ratio for x and y axes
plt.xlim(-1, 1)  # Set the x-axis limits
plt.ylim(-1, 1)  # Set the y-axis limits
ax.set_xticks((-1, 0, 1))  # Set x-axis ticks at -1, 0, and 1
ax.set_yticks((-1, 0, 1))  # Set y-axis ticks at -1, 0, and 1

# (2) Display information and calculate the estimate of π
st.write('Number of points inside = ' + str(mask_inside.sum()))  # Show the number of points inside the unit circle
st.write('Percentage of points inside = ' + str(mask_inside.sum() / num * 100) + '%')  # Show the percentage of points inside
estimated_pi = mask_inside.sum() / num * 4  # Estimate π using the Monte Carlo method
st.write('Estimated pi = ' + str(np.round(estimated_pi, 5)))  # Display the estimated value of π

st.pyplot(fig)  # (3) Render the Matplotlib figure in the Streamlit app
st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)
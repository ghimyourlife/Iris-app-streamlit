# import libraries
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  
import numpy as np  
from sklearn import datasets 
from sklearn.mixture import GaussianMixture 
import streamlit as st 
from matplotlib.patches import Ellipse  

# Configure Matplotlib parameters for styling
p = plt.rcParams
p["font.sans-serif"] = ["DejaVu Sans"]  # Set font style to DejaVu Sans
p["font.weight"] = "light"  # Set font weight to light
p["ytick.minor.visible"] = True  # Enable visibility of minor ticks on the y-axis
p["xtick.minor.visible"] = True  # Enable visibility of minor ticks on the x-axis
p["axes.grid"] = True  # Enable grid lines on the plot
p["grid.color"] = "0.5"  # Set grid line color to gray
p["grid.linewidth"] = 0.5  # Set grid line width

# Define a function to draw ellipses for GMM clusters
def make_ellipses(gmm, ax):
    for j in range(0, K):  # Iterate through the number of clusters
        # Handle different covariance types
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[j]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[j])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[j]

        U, S, V_T = np.linalg.svd(covariances)  # Perform Singular Value Decomposition
        major, minor = 2 * np.sqrt(S)  # Compute lengths of major and minor axes
        angle = np.arctan2(U[1, 0], U[0, 0]) * 180 / np.pi  # Compute rotation angle in degrees
        
        # Plot the cluster center
        ax.plot(gmm.means_[j, 0], gmm.means_[j, 1], color='k', marker='x', markersize=10)

        # Draw the semi-major and semi-minor axes
        ax.quiver(gmm.means_[j, 0], gmm.means_[j, 1], U[0, 0], U[1, 0], scale=5 / major)
        ax.quiver(gmm.means_[j, 0], gmm.means_[j, 1], U[0, 1], U[1, 1], scale=5 / minor)

        # Draw ellipses for different scales
        for scale in np.array([3, 2, 1]):
            #ell = Ellipse(gmm.means_[j, :2], scale * major, scale * minor, angle, 
                          color=rgb[j, :], alpha=0.18)
             ell = Ellipse(xy=gmm.means_[j, :2],  # Center of the ellipse
                          width=scale * major,   # Total width (major axis length)
                          height=scale * minor,  # Total height (minor axis length)
                          angle=angle,           # Rotation angle
                          facecolor=rgb[j, :],   # Face color for the ellipse
                          edgecolor='none',      # No edge color
                          alpha=0.18) 
            ax.add_artist(ell)

x1_array = np.linspace(4, 8, 101)  # Generate evenly spaced values for x1
x2_array = np.linspace(1, 5, 101)  # Generate evenly spaced values for x2
xx1, xx2 = np.meshgrid(x1_array, x2_array)  # Create a 2D grid for x1 and x2

iris = datasets.load_iris()
X = iris.data[:, :2]  # Select the first two features for clustering

covariance_types = ['tied', 'spherical', 'diag', 'full']

# Create Streamlit sidebar
with st.sidebar:
    st.title('GMM Clustering')  # Add title to the sidebar
    covariance_type = st.radio('covariance_type', covariance_types)  # Radio buttons for covariance types

K = 3  # Set the number of clusters to 3

rgb = [[255, 51, 0], [0, 153, 255], [138, 138, 138]]  # Define RGB colors for clusters
rgb = np.array(rgb) / 255.  # Normalize color values to [0, 1]
cmap_bold = ListedColormap(rgb)  # Create a colormap for clusters

# Apply GMM clustering
gmm = GaussianMixture(n_components=K, covariance_type=covariance_type)  # Initialize GMM with chosen covariance type
gmm.fit(X)  # Train the GMM on the dataset
Z = gmm.predict(np.c_[xx1.ravel(), xx2.ravel()])  # Predict cluster labels for grid points
Z = Z.reshape(xx1.shape)  # Reshape predictions to match grid shape

# Visualization
fig = plt.figure(figsize=(10, 5))  # Create a figure with specified size
ax = fig.add_subplot(1, 2, 1)  # Add the first subplot
ax.scatter(x=X[:, 0], y=X[:, 1], color=np.array([0, 68, 138]) / 255., 
           alpha=1.0, linewidth=1, edgecolor=[1, 1, 1])  # Scatter plot of data points
make_ellipses(gmm, ax)  # Draw ellipses and vectors for clusters
ax.set_xlim(4, 8)  # Set x-axis limits
ax.set_ylim(1, 5)  # Set y-axis limits
ax.set_xlabel(iris.feature_names[0])  # Label x-axis
ax.set_ylabel(iris.feature_names[1])  # Label y-axis
ax.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])  # Add a dashed grid
ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal

ax = fig.add_subplot(1, 2, 2)  # Add the second subplot
ax.contourf(xx1, xx2, Z, cmap=cmap_bold, alpha=0.18)  # Plot filled contours for cluster regions
ax.contour(xx1, xx2, Z, levels=[0, 1, 2], colors=[np.array([0, 68, 138]) / 255.])  # Add contour lines
ax.scatter(x=X[:, 0], y=X[:, 1], color=np.array([0, 68, 138]) / 255., 
           alpha=1.0, linewidth=1, edgecolor=[1, 1, 1])  # Scatter plot of data points
centroids = gmm.means_  # Extract cluster centers
ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, linewidths=1.5, color="k")  # Plot cluster centers
ax.set_xlim(4, 8)  # Set x-axis limits
ax.set_ylim(1, 5)  # Set y-axis limits
ax.set_xlabel(iris.feature_names[0])  # Label x-axis
ax.set_ylabel(iris.feature_names[1])  # Label y-axis
ax.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])  # Add a dashed grid
ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal

st.pyplot(fig)  # Render the plot in the Streamlit app
st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)
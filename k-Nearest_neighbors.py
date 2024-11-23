# import libraries
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns  
from matplotlib.colors import ListedColormap  
from sklearn import neighbors, datasets  
import streamlit as st  

# Configure Matplotlib parameters
p = plt.rcParams
p["font.sans-serif"] = ["DejaVu Sans"]  # Set font style to DejaVu Sans
p["font.weight"] = "light"  # Set font weight to light
p["ytick.minor.visible"] = True  # Enable minor ticks on the y-axis
p["xtick.minor.visible"] = True  # Enable minor ticks on the x-axis
p["axes.grid"] = True  # Enable grid lines
p["grid.color"] = "0.5"  # Set grid color to medium gray
p["grid.linewidth"] = 0.5  # Set grid line width

# Sidebar setup in Streamlit
with st.sidebar:
    st.title('k-Nearest Neighbors Classification')  # Add a title to the sidebar
    k_neighbors = st.slider('k_neighbors',  # Slider to select the number of neighbors (k)
                            min_value=2, 
                            max_value=20, 
                            value=5, step=1)

# Load and preprocess the Iris dataset
iris = datasets.load_iris()  # Load the Iris dataset
X = iris.data[:, :2]  # Select the first two features for visualization
y = iris.target  # Target labels (classes)

# Generate a grid of points for decision boundary visualization
x1_array = np.linspace(4, 8, 101)  # Generate x1 values for the grid
x2_array = np.linspace(1, 5, 101)  # Generate x2 values for the grid
xx1, xx2 = np.meshgrid(x1_array, x2_array)  # Create a mesh grid of points

# Define color maps for visualization
rgb = [[255, 238, 255],  # Light pink for background
       [219, 238, 244],  # Light blue for background
       [228, 228, 228]]  # Light gray for background
rgb = np.array(rgb)/255.  # Normalize color values
cmap_light = ListedColormap(rgb)  # Create a colormap for light background
cmap_bold = [[255, 51, 0],  # Red for class 1
             [0, 153, 255],  # Blue for class 2
             [138, 138, 138]]  # Gray for class 3
cmap_bold = np.array(cmap_bold)/255.  # Normalize bold color values

# Create and train the kNN classifier
kNN = neighbors.KNeighborsClassifier(k_neighbors)  # Initialize the kNN classifier with the selected k
kNN.fit(X, y)  # Train the kNN classifier using the dataset

# Predict class labels for the grid points
q = np.c_[xx1.ravel(), xx2.ravel()]  # Reshape the grid points into query points
y_predict = kNN.predict(q)  # Predict class labels for the query points
y_predict = y_predict.reshape(xx1.shape)  # Reshape predictions back to grid shape

# Visualization of decision boundaries and data points
fig, ax = plt.subplots()  # Create a Matplotlib figure and axis
plt.contourf(xx1, xx2, y_predict, cmap=cmap_light)  # Plot the decision boundaries with light background colors
plt.contour(xx1, xx2, y_predict, levels=[0, 1, 2],  # Add contour lines for class boundaries
            colors=np.array([0, 68, 138])/255.)  # Set contour line color to dark blue
sns.scatterplot(x=X[:, 0], y=X[:, 1],  # Scatter plot of the original data points
                hue=iris.target_names[y],  # Use target names for legend
                ax=ax, 
                palette=dict(setosa=cmap_bold[0, :],  # Red for Setosa
                             versicolor=cmap_bold[1, :],  # Blue for Versicolor
                             virginica=cmap_bold[2, :]),  # Gray for Virginica
                alpha=1.0,  # Full opacity for data points
                linewidth=1, edgecolor=[1, 1, 1])  # White edge around points for visibility
plt.xlim(4, 8)  # Set x-axis limits
plt.ylim(1, 5)  # Set y-axis limits
plt.xlabel(iris.feature_names[0])  # Label the x-axis with the first feature name
plt.ylabel(iris.feature_names[1])  # Label the y-axis with the second feature name
ax.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])  # Add a dashed grid for better readability
ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio to equal

# Display the plot in the Streamlit app
st.pyplot(fig)  # Render the Matplotlib figure in the Streamlit app
st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)
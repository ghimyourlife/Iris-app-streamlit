# import libraries
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import seaborn as sns  # Import Seaborn for enhanced visualizations
from matplotlib.colors import ListedColormap  # Import for custom color maps
from sklearn import neighbors, datasets  # Import neighbors and datasets for kNN and Iris dataset
import streamlit as st  # Import Streamlit for building interactive apps

# Configure Matplotlib parameters for styling
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]  # Set font style to Roboto
p["font.weight"] = "light"  # Set font weight to light
p["ytick.minor.visible"] = True  # Enable minor ticks on the y-axis
p["xtick.minor.visible"] = True  # Enable minor ticks on the x-axis
p["axes.grid"] = True  # Enable grid lines on plots
p["grid.color"] = "0.5"  # Set grid line color
p["grid.linewidth"] = 0.5  # Set grid line width

# Sidebar setup for user input in Streamlit
with st.sidebar:
    st.title('Support Vector Machine')  # Add title to the sidebar
    st.write('Gaussian kernel')  # Label for Gaussian kernel parameter
    gamma = st.slider('Gamma',  # Add a slider for selecting gamma parameter
                      min_value=0.001,  # Minimum value for gamma
                      max_value=5.0,  # Maximum value for gamma
                      value=1.0,  # Default value for gamma
                      step=0.05)  # Step size for gamma

# Load and preprocess data
iris = datasets.load_iris()  # Load the Iris dataset
X = iris.data[:, :2]  # Select the first two features (sepal length and sepal width)
y = iris.target  # Extract the class labels

# Generate grid data for decision boundary visualization
x1_array = np.linspace(4, 8, 101)  # Generate 101 evenly spaced values between 4 and 8 for x1
x2_array = np.linspace(1, 5, 101)  # Generate 101 evenly spaced values between 1 and 5 for x2
xx1, xx2 = np.meshgrid(x1_array, x2_array)  # Create a 2D grid of x1 and x2 values

# Define color maps for visualization
rgb = [[255, 238, 255],  # Light pink for the background
       [219, 238, 244],  # Light blue for the background
       [228, 228, 228]]  # Light gray for the background
rgb = np.array(rgb) / 255.  # Normalize color values to the range [0, 1]
cmap_light = ListedColormap(rgb)  # Create a colormap for light background regions

cmap_bold = [[255, 51, 0],  # Red for class 1
             [0, 153, 255],  # Blue for class 2
             [138, 138, 138]]  # Gray for class 3
cmap_bold = np.array(cmap_bold) / 255.  # Normalize bold color values to [0, 1]

# Reshape grid points into query points
q = np.c_[xx1.ravel(), xx2.ravel()]  # Flatten and stack xx1 and xx2 into a 2D array of query points

# Create an SVM classifier with a Gaussian (RBF) kernel
from sklearn import svm  # Import SVM module from scikit-learn
SVM = svm.SVC(kernel='rbf', gamma=gamma)  # Initialize SVM with RBF kernel and user-defined gamma
SVM.fit(X, y)  # Train the SVM model on the dataset (X, y)

# Predict class labels for all query points
y_predict = SVM.predict(q)  # Use the trained SVM model to predict labels for query points
y_predict = y_predict.reshape(xx1.shape)  # Reshape the predictions to match the grid shape

# Visualization of decision boundaries and data points
fig, ax = plt.subplots()  # Create a Matplotlib figure and axis
plt.contourf(xx1, xx2, y_predict, cmap=cmap_light)  # Plot decision boundaries with light background colors
plt.contour(xx1, xx2, y_predict, levels=[0, 1, 2],  # Add contour lines at class boundaries
            colors=np.array([0, 68, 138]) / 255.)  # Set contour line color to dark blue

sns.scatterplot(x=X[:, 0], y=X[:, 1],  # Scatter plot of the original data points
                hue=iris.target_names[y],  # Color points based on class labels
                ax=ax,  # Specify the axis to plot on
                palette=dict(setosa=cmap_bold[0, :],  # Red for Setosa
                             versicolor=cmap_bold[1, :],  # Blue for Versicolor
                             virginica=cmap_bold[2, :]),  # Gray for Virginica
                alpha=1.0,  # Full opacity for data points
                linewidth=1, edgecolor=[1, 1, 1])  # Set edge color for scatter points

# Set axis limits and labels
plt.xlim(4, 8)  # Set x-axis limits
plt.ylim(1, 5)  # Set y-axis limits
plt.xlabel(iris.feature_names[0])  # Label x-axis with the first feature name
plt.ylabel(iris.feature_names[1])  # Label y-axis with the second feature name

# Configure grid and aspect ratio
ax.grid(linestyle='--', linewidth=0.25,  # Add a dashed grid with light gray lines
        color=[0.5, 0.5, 0.5])  # Grid line color
ax.set_aspect('equal', adjustable='box')  # Set equal aspect ratio for the axis

st.pyplot(fig)  # Render the plot in the Streamlit app
st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)
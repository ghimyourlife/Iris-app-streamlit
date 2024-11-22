import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for visualization
from sklearn.preprocessing import PolynomialFeatures  # Import PolynomialFeatures for polynomial regression
from sklearn.linear_model import LinearRegression  # Import LinearRegression for training the model
import streamlit as st  # Import Streamlit for building the app

# Configure Matplotlib plot parameters
p = plt.rcParams
p["font.sans-serif"] = ["DejaVu Sans"]  # Set the default font to DejaVu Sans
p["font.weight"] = "light"  # Set font weight to light
p["ytick.minor.visible"] = True  # Enable minor ticks on the y-axis
p["xtick.minor.visible"] = True  # Enable minor ticks on the x-axis
p["axes.grid"] = True  # Enable grid lines on the plot
p["grid.color"] = "0.5"  # Set grid line color to medium gray
p["grid.linewidth"] = 0.5  # Set grid line width

# Generate random data
np.random.seed(0)  # Set random seed for reproducibility
num = 30  # Number of data points
X = np.random.uniform(0, 4, num)  # Generate random x values between 0 and 4
y = np.sin(0.4 * np.pi * X) + 0.4 * np.random.randn(num)  # Generate noisy sine wave data
data = np.column_stack([X, y])  # Combine x and y into a single array

x_array = np.linspace(0, 4, 101).reshape(-1, 1)  # Generate evenly spaced x values for predictions
degree_array = [1, 2, 3, 4, 7, 8]  # Define the polynomial degrees for reference

# Create a sidebar for user input
with st.sidebar:
    st.title('Polynomial Regression')  # Add a title to the sidebar
    degree = st.slider('Degree', min_value=1, max_value=9, value=2, step=1)  # Slider to select the polynomial degree

# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(5, 5))  # Initialize a figure with specified size

poly = PolynomialFeatures(degree=degree)  # Generate polynomial features of the selected degree
X_poly = poly.fit_transform(X.reshape(-1, 1))  # Transform the original data into polynomial features

# Train a linear regression model
poly_reg = LinearRegression()  # Initialize a linear regression model
poly_reg.fit(X_poly, y)  # Fit the model to the polynomial features and target values
y_poly_pred = poly_reg.predict(X_poly)  # Predict y values for the training data
data_ = np.column_stack([X, y_poly_pred])  # Combine x and predicted y values for visualization

y_array_pred = poly_reg.predict(  # Predict y values for the evenly spaced x values
    poly.fit_transform(x_array))

# Plot scatter points
ax.scatter(X, y, s=20)  # Scatter plot of the original data points
ax.scatter(X, y_poly_pred, marker='x', color='k')  # Scatter plot of predicted values as black x markers

ax.plot(([i for (i, j) in data_], [i for (i, j) in data]),  # Plot lines connecting actual and predicted points
        ([j for (i, j) in data_], [j for (i, j) in data]),
        c=[0.6, 0.6, 0.6], alpha=0.5)  # Light gray lines with some transparency

ax.plot(x_array, y_array_pred, color='r')  # Plot the predicted regression line in red

# Extract model parameters
coef = poly_reg.coef_  # Extract the coefficients of the polynomial
intercept = poly_reg.intercept_  # Extract the intercept of the model

# Construct the regression equation
equation = '$y = {:.1f}'.format(intercept)  # Start the equation with the intercept
for j in range(1, len(coef)):  # Loop through the coefficients
    equation += ' + {:.1f}x^{}'.format(coef[j], j)  # Add each term of the polynomial
equation += '$'  # Close the equation
equation = equation.replace("+ -", "-")  # Replace "+ -" with "-" for cleaner formatting

st.write(equation)  # Display the regression equation in the app
ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio of the plot to equal
ax.set_xlim(0, 4)  # Set x-axis limits
ax.grid(False)  # Disable the grid for the plot
ax.set_ylim(-2, 2)  # Set y-axis limits

st.pyplot(fig)  # Render the plot using Streamlit

#st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)
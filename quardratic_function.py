# import modules
import streamlit as st  
import numpy as np  
from sympy import symbols, lambdify  
import matplotlib.pyplot as plt  

# Sidebar
with st.sidebar:  # a. Define a sidebar block
    st.header('Choose coefficients')  # Add a header in the sidebar
    st.latex(r'f(x) = ax^2 + bx + c')  # b. Display the quadratic function formula in LaTeX format
    a = st.slider("a", min_value=-5.0, max_value=5.0,
                   step=0.01, value=1.0)  # c. Slider for coefficient 'a'
    b = st.slider("b", min_value=-5.0, max_value=5.0,
                   step=0.01, value=-2.0)  # Slider for coefficient 'b'
    c = st.slider("c", min_value=-5.0, max_value=5.0,
                   step=0.01, value=-3.0)  # Slider for coefficient 'c'

# Quadratic function
x = symbols('x')  # Define 'x' as a symbolic variable
f_x = a * x**2 + b * x + c  # d. Define the quadratic function f(x)
x_array = np.linspace(-5, 5, 101)  # Generate an array of x values from -5 to 5
f_x_fcn = lambdify(x, f_x)  # e. Convert the symbolic function f(x) to a Python function
y_array = f_x_fcn(x_array)  # f. Evaluate f(x) for all x values in x_array

# Main page
st.title('Quadratic function')  # g. Add a title to the main page
st.latex(r'f(x) = ')  # h. Display the quadratic function in LaTeX format
st.latex(f_x)  # Display the specific quadratic function based on chosen coefficients

# Visualization
fig = plt.figure()  # Create a Matplotlib figure
ax = fig.add_subplot(111)  # Add a subplot to the figure
ax.plot(x_array, y_array)  # Plot the quadratic function

ax.set_xlim([-5, 5])  # Set the x-axis limits
ax.set_ylim([-5, 5])  # Set the y-axis limits
ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio to equal
ax.set_xlabel('x')  # Label the x-axis
ax.set_ylabel('f(x)')  # Label the y-axis
st.write(fig)  # i. Display the Matplotlib figure in the Streamlit app

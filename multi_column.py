import streamlit as st  # Import Streamlit for building the app

# Create two columns
col1, col2 = st.columns(2)  # a. Create a two-column layout
col1.write("This is column 1")  # b. Display text in the first column
col1.latex(r'f(x) = ax^2 + bx + c')  # Display a LaTeX formula in the first column

col2.write("This is column 2")  # Display text in the second column
# import libraries
import plotly.express as px  # Import Plotly Express for creating visualizations
import streamlit as st  # Import Streamlit for building interactive web apps

# Load the Iris dataset
df = px.data.iris()  # Load the built-in Iris dataset from Plotly Express
features = df.columns[:4]  # Extract the first four columns as potential features (sepal and petal measurements)

# Create a sidebar for user input
with st.sidebar:  # Define the sidebar in the Streamlit app
    st.title('Iris Dataset')  # Add a title to the sidebar
    x_col = st.radio('X-axis', features)  # Add a radio button to select the column for the X-axis
    y_col = st.radio('Y-axis', features)  # Add a radio button to select the column for the Y-axis
    marginal_x = st.radio('X-axis Marginal Plot', ["histogram", "rug", "box", "violin"])  # Choose marginal plot type for X-axis
    marginal_y = st.radio('Y-axis Marginal Plot', ["histogram", "rug", "box", "violin"])  # Choose marginal plot type for Y-axis

# Create a scatter plot using Plotly Express
fig = px.scatter(df, 
                 x=x_col,  # Set the selected column as the X-axis
                 y=y_col,  # Set the selected column as the Y-axis
                 color="species",  # Color the points by the species of the iris
                 marginal_x=marginal_x,  # Add a marginal plot to the X-axis
                 marginal_y=marginal_y,  # Add a marginal plot to the Y-axis
                 width=650,  # Set the width of the plot
                 height=600)  # Set the height of the plot

# Display the plot in the Streamlit app
st.plotly_chart(fig)  # Render the Plotly chart in the app
#st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)
import streamlit as st  # a. Import the Streamlit library for building the app
import seaborn as sns  # Import Seaborn to load and handle the Iris dataset
import plotly.express as px  # Import Plotly Express for creating visualizations

st.title('Welcome to the world of :red[Streamlit]')  # b. Display the app title with "Streamlit" in red
st.header('Pandas DataFrame')  # c. Add a header to introduce the Pandas DataFrame section
st.markdown("Load :blue[Iris Data Set]")  # d. Display a markdown text with "Iris Data Set" in blue

df = sns.load_dataset('iris')  # Load the Iris dataset from Seaborn as a Pandas DataFrame
st.write(df)  # e. Display the DataFrame in the app

st.header('Visualize Using Heatmap')  # Add a header to introduce the heatmap section
fig = px.imshow(df.iloc[:, :-1])  # Create a heatmap using the numerical columns of the dataset
st.write(fig)  # Display the heatmap in the app
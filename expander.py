import streamlit as st 
import seaborn as sns  
import plotly.express as px  

st.title('Iris Dataset')  # Set the app title as "Iris Dataset"

df = sns.load_dataset('iris')  # Load the Iris dataset into a DataFrame

# First expandable section
with st.expander("Open and view DataFrame"):  # Create an expandable section titled "Open and view DataFrame"
    st.write(df)  # Display the Iris dataset as a DataFrame inside the expandable section

# Second expandable section
with st.expander("Open and view Heatmap"):  # Create an expandable section titled "Open and view Heatmap"
    fig = px.imshow(df.iloc[:, :-1])  # Create a heatmap using the numerical columns of the Iris dataset
    st.write(fig)  # Display the heatmap inside the expandable section
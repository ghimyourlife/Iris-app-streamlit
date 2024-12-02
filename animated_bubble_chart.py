# import libraries
import plotly.express as px  
import streamlit as st 

df = px.data.gapminder()  # Load the built-in Gapminder dataset from Plotly Express

# Create a sidebar in Streamlit
with st.sidebar:  
    st.title('Bubble Chart')  # Add a title to the sidebar

# Create an animated bubble chart using Plotly Express
fig = px.scatter(df, 
                 x="gdpPercap",  # Set GDP per capita as the X-axis
                 y="lifeExp",  # Set life expectancy as the Y-axis
                 animation_frame="year",  # Create animation frames based on the year column
                 animation_group="country",  # Group animation by country
                 size="pop",  # Set the size of the bubbles based on population
                 color="continent",  # Color the bubbles based on the continent
                 hover_name="country",  # Show the country name when hovering over a bubble
                 log_x=True,  # Use a logarithmic scale for the X-axis
                 size_max=55,  # Set the maximum size of the bubbles
                 range_x=[100, 100000],  # Set the range for the X-axis
                 range_y=[25, 90])  # Set the range for the Y-axis

st.plotly_chart(fig)  # Render the Plotly figure in the app
st.markdown('<p style="text-align: center; color: gray;">Code download please visit <a href="https://github.com/visualize-ml" target="_blank">Github Repo: Visualize-ML</a></p>', unsafe_allow_html=True)
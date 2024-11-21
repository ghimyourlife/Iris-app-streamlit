import streamlit as st 

# Create two tabs with different content
tab_A, tab_B = st.tabs(["Tab A", "Tab B"])  # Create two tabs named "Tab A" and "Tab B"

with tab_A:  # Define the content for Tab A
   st.header("Tab A Title")  # Add a header for Tab A
   st.write('This is Tab A.')  # Add some text to Tab A

with tab_B:  # Define the content for Tab B
   st.header("Tab B Title")  # Add a header for Tab B
   st.write('This is Tab B.')  # Add some text to Tab B
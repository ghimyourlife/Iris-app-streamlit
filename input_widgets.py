import streamlit as st  
button_return = st.button("Click me")  # Display a button widget, returns True when clicked
st.write(button_return)  # Display the return value of the button (True or False)
st.checkbox("Check me")  # Display a checkbox widget
st.radio("Choose one:",
         ["A", "B", "C"])  # Display a radio button widget for selecting one option
st.selectbox("Choose one:",
             ["A", "B", "C"])  # Display a dropdown widget for selecting one option
st.multiselect("Choose many:",
               ["A", "B", "C", "D"])  # Display a multiselect widget for selecting multiple options
st.slider("Select a value:",
           0.0, 10.0, 5.0)  # Display a slider widget for selecting a value within a range
st.select_slider("Select a value:",
                  options=[1, 2, 3, 4, 5])  # Display a slider widget with discrete options
st.text_input("Enter your name")  # Display a text input widget for entering text
st.number_input("Enter a number")  # Display a numeric input widget for entering a number
st.text_area("Enter your message")  # Display a text area widget for entering multi-line text
st.date_input("Select a date")  # Display a date input widget for selecting a date
st.time_input("Select a time")  # Display a time input widget for selecting a time
st.file_uploader("Upload a file")  # Display a file uploader widget for uploading files
st.color_picker("Pick a color")  # Display a color picker widget for selecting a color
import streamlit as st
from io import BytesIO
import time
import pandas as pd
import pages.spectrogram as s
#import pages
# from pages import spectrogram
# from spectro
# from page
# from spectrogram import draw_spectrogram_from_dataframe
#from pages.spectrogram import draw_spectrogram_from_dataframe


def apply_custom_css(css):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

custom_css = """
    body {
        background-color: #FFFFFF;
    }
    .main {
        background-color: #7D33F2;
    }
    .st-bd .st-bs .st-cs .st-ck  {
        background-color: #757575;
    }
    h1 {
        color: #FFFFFF;
    }
"""

apply_custom_css(custom_css)



def run_progress_bar():
    progress_bar = st.progress(0)
    for i in range(101):
        time.sleep(0.01)
        progress_bar.progress(i)


st.sidebar.title('Navigation')



st.title("Wi-Fi Activity Recognition")


# Create a file upload button for multiple files
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["csv"])

# Store the uploaded files in a dictionary with unique keys
uploaded_files_dict = {f"File {i + 1}": f for i, f in enumerate(uploaded_files)}


# Process the uploaded files (e.g., read the content)
if uploaded_files_dict:
    for file_key, file_obj in uploaded_files_dict.items():
        # Read the file content as bytes
        file_content = BytesIO(file_obj.getvalue())
        # Process the file content here
    start_button = st.button("Start")
    if start_button:
        for uploaded_file in uploaded_files:
            st.write(f"Processing {uploaded_file.name}")
            progress_bar = st.progress(0)

            # Simulate a long-running operation when reading the CSV file
            for i in range(101):
                time.sleep(0.01)
                progress_bar.progress(i)

            df = pd.read_csv(uploaded_file, header=None)
            df2 = pd.read_csv(uploaded_file, header=None).values
            st.write(df.iloc[:5, :10])
            st.write("----" * 20)  # Add a separator line between headers
            
            st.write("Summary of uploaded data:")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")

            s.visualize(df2)
            progress_bar.empty()

else:
    st.write("No files uploaded.")

#start_button = st.button("Start")

st.subheader("Model Inference")

# Define the options for the dropout menu
dropout_options = ['MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN', 'GRU', 'LSTM', 'BiLSTM', 'CNN+GRU', 'ViT']

# Create a dropdown menu using st.selectbox
dropout_rate = st.selectbox('Select the model for inference:', dropout_options)


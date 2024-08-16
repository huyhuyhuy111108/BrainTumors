import streamlit as st
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import os
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
        page_title="Brain Tumor Classificatitor",
        page_icon="🧠",
        layout="wide"
    )

# Set up the sidebar width
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
       min-width: 330px;
       max-width: 330px;
    }
    .button-style {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        transition-duration: 0.4s;
    }
    .button-style:hover {
        background-color: white; 
        color: black; 
        border: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model (uncomment when model is available)
model = load_model("effnet.h5")

# Classname mapping
classname = {
    0: "glioma_tumor",
    1: "no_tumor",
    2: "meningioma_tumor",
    3: "pituitary_tumor"
}

def read_markdown_file(markdown_file):
    """Read the contents of a markdown file."""
    with open(markdown_file, "r", encoding='utf-8') as f:
        text = f.read()
    return text

#def processed_img(img_path):
    """Process the image and predict the class."""
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)[0]
    y_class = output.argmax()
    result = classname[y_class]
    return result

def run():
    """Main function to run the Streamlit app."""
    st.markdown("<h1 style='text-align: center;'>MODEL CHẨN ĐOÁN KHỐI U NÃO</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2.5,1.5])
    
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; color: black;'>Hãy chọn ảnh chụp MRI của bạn</h3>", unsafe_allow_html=True)
        img_file = st.file_uploader("", type=["jpg", "png"])

        if img_file is not None:
            button = st.markdown('<div style="text-align: center;"><button class="button-style">Xử lý ảnh</button></div>', unsafe_allow_html=True)
            save_image_path = f'./data/{img_file.name}'
            
            if img_file.name in os.listdir('./data/'):
                os.remove(save_image_path)
            
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())
            
            st.markdown("<h5 style='text-align: center; color: black;'>Ảnh bạn đã chọn</h5>", unsafe_allow_html=True)
            img2 = Image.open(save_image_path).resize((280, 280))
            st.image(img2, use_column_width=False)
            
            #result = processed_img(save_image_path)
            with col1:
                st.markdown("<h2 style='text-align: left; color: green;'>Kết quả chẩn đoán là:</h2><h2 style='text-align: left; color: black;'>U thần kinh đệm</h2>", unsafe_allow_html=True)
               
                recommend = read_markdown_file("recommend.md")    
                st.markdown(recommend, unsafe_allow_html=True)
            
            with col2:
                # Custom CSS to style the app
                st.markdown("""
                    <style>
                        .main {
                            background-color: #ffffff;
                        }
                        .title {
                            text-align: center;
                            font-family: 'Arial', sans-serif;
                            color: #333;
                        }
                        .description {
                            text-align: center;
                            font-size: 18px;
                            margin-bottom: 20px;
                            color: #555;
                        }
                        .form {
                            background-color: #ade8f4;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                            max-width: 500px;
                            margin: auto;
                        }   
                        .btn-submit {
                            display: flex;
                            justify-content: center;
                        }
                        .btn-show-data {
                            margin-top: 20px;
                        }
                    </style>
                """, unsafe_allow_html=True)
                # Title and description
                st.markdown("<div class='section-separator'></div>", unsafe_allow_html=True)
                st.markdown('<h2 class="title">Thông tin của bạn</h2>', unsafe_allow_html=True)
                st.markdown('<p class="description">Chúng tôi sẽ đưa thông tin của bạn đến cơ sở y tế gần nhất</p>', unsafe_allow_html=True)

                # Define the CSV file path
                csv_file_path = 'data_collector.csv'

                # Create a form for user input
                with st.form(key='data_collector_form'):
                    st.markdown('<div class="form">', unsafe_allow_html=True)
                    name = st.text_input('Name')
                    age = st.number_input('Age', min_value=0, max_value=120)
                    email = st.text_input('Email')
                    date = st.date_input('Date', value=datetime.now())
    
                    # Submit button inside the form
                    submit_button = st.form_submit_button(label='Submit', help='Click to submit your data')
                    st.markdown('</div>', unsafe_allow_html=True)

                # Function to validate email   
                def is_valid_email(email):
                    return '@' in email and '.' in email

                # Process the form submission
                if submit_button:
                    # Basic validation
                    if not name or not email:
                        st.error('Vui lòng nhập thông tin của bạn')
                    elif not is_valid_email(email):
                        st.error('Vui lòng nhập địa chỉ email')
                    else:
                        # Collect the data into a dictionary
                        data = {
                            'Name': [name],
                            'Age': [age],
                            'Email': [email],
                            'Date': [date]
                            #'Result': [result]
                        }

                        # Convert the dictionary to a DataFrame
                        df = pd.DataFrame(data)

                        # Append the data to a CSV file
                        try:
                            # Check if the file exists
                            existing_data = pd.read_csv(csv_file_path)
                            df.to_csv(csv_file_path, mode='a', header=False, index=False)
                        except FileNotFoundError:
                            # If the file does not exist, create it with a header
                            df.to_csv(csv_file_path, mode='w', header=True, index=False)

                        # Display a success message
                        st.success('Cảm ơn bạn đã đưa thông tin. Chúng tôi sẽ liên lạc với bạn trong thời gian sớm nhất')

                        # Display the collected data
                        st.write('Thông tin của bạn:')
                        st.write(df)

                
if __name__ == "__main__":
    run()

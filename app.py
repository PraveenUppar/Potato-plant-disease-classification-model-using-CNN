import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings

filterwarnings('ignore')


def streamlit_config():
    # Page configuration
    st.set_page_config(page_title='Potato Disease Classifier', layout='centered')

    # Page header transparent color and custom CSS
    page_background_color = """
    <style>
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    .main-header {
        text-align: center;
        color: #2C3E50; /* Darker tone for a professional look */
        font-size: 3.5em;
        font-weight: bold;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .tagline {
        text-align: center;
        color: #6C757D;
        font-size: 1.2em;
        margin-top: 0px;
    }
    .section-header {
        color: #2C3E50;
        font-size: 2em;
        border-bottom: 2px solid #FF4B4B; /* Keeping the accent color for underlines */
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #4CAF50; /* Green for action buttons */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.1em;
    }
    .prediction-text {
        color: #FF4B4B; /* Accent color for prediction */
        font-size: 1.8em;
        font-weight: bold;
        text-align: center;
    }
    .info-box {
        background-color: #E6F3F7;
        border-left: 5px solid #007BFF;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)


# Streamlit Configuration Setup
streamlit_config()


def prediction(image_path, class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    img = Image.open(image_path)
    img_resized = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Correct the model path - assuming 'model.h5' is in the same directory
    model_path = 'model.h5'
    
    try:
        model = tf.keras.models.load_model(model_path)
        prediction_result = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction_result)]
        confidence = round(np.max(prediction_result) * 100, 2)

        st.markdown(f'<p class="prediction-text">Predicted Class: {predicted_class}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: center; font-size: 1.2em;">Confidence: {confidence}%</p>', unsafe_allow_html=True)
        add_vertical_space(1)
        st.image(img.resize((400, 300)), caption='Uploaded Image', use_column_width=False)
        st.success("Prediction successful!")

    except Exception as e:
        st.error(f"Error loading model or making prediction: {e}")
        st.warning("Please ensure 'model.h5' is in the correct directory relative to your application script.")


# --- Hero Section ---
st.markdown(f'<h1 class="main-header">Potato Disease Classification</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="tagline">An AI-powered solution for early detection in potato plants</p>', unsafe_allow_html=True)
add_vertical_space(2)


# --- Project Overview Section ---
st.markdown(f'<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
st.markdown("""
In agriculture, identifying potato plant diseases like early blight, late blight, or determining plant health is crucial for effective crop management. This project addresses these challenges by applying **deep learning** to enable **real-time disease classification**, helping farmers optimize resource allocation and enhance crop yield.

This **web application**, powered by a **Convolutional Neural Network (CNN)**, classifies potato plant diseases (healthy, late blight, early blight) to facilitate early intervention and prevention. The model was trained on a **9,000-image dataset** (70/30 train-test split) and achieved significant performance improvements:

* **Accuracy:** Improved from 66.8% to **89.2%**.
* **Training Efficiency:** Epochs reduced from 9 to **5**, cutting training time by **45%** (from 20 to 11 minutes) while preventing overfitting.

The core technologies employed include **Python, TensorFlow, Keras, NumPy, Matplotlib, and Streamlit**. The dataset was sourced from Kaggle.
""")
add_vertical_space(3)


# --- Image Uploader Section ---
st.markdown(f'<h2 class="section-header">Upload Image for Prediction</h2>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([0.1, 0.9, 0.1])
with col2:
    input_image = st.file_uploader(label='Choose a Potato Leaf Image', type=['jpg', 'jpeg', 'png'])

if input_image is not None:
    st.info("Processing your image...")
    add_vertical_space(1)
    col_pred1, col_pred2, col_pred3 = st.columns([0.2, 0.8, 0.2])
    with col_pred2:
        prediction(input_image)
else:
    st.info("Upload an image to classify the potato plant disease (Early Blight, Late Blight, or Healthy).")

add_vertical_space(5)
st.markdown("---")
st.markdown(f'<p style="text-align: center; color: #888888; font-size: 0.9em;">Developed using Streamlit and TensorFlow.</p>', unsafe_allow_html=True)

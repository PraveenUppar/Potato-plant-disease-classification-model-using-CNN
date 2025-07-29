import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings

filterwarnings('ignore')


def streamlit_config():
    # page configuration
    st.set_page_config(page_title='Potato Disease Classifier', layout='centered')

    # page header transparent color
    page_background_color = """
    <style>
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    .main-header {
        text-align: center;
        color: #FF4B4B; /* Streamlit's default orange for accent */
        font-size: 3.5em; /* Larger font size for main title */
        font-weight: bold;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .tagline {
        text-align: center;
        color: #6C757D; /* A subtle grey for tagline */
        font-size: 1.2em;
        margin-top: 0px;
    }
    .section-header {
        color: #2C3E50; /* Darker color for section headers */
        font-size: 2em;
        border-bottom: 2px solid #FF4B4B; /* Underline with accent color */
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.1em;
    }
    .prediction-text {
        color: #FF4B4B; /* Orange for prediction */
        font-size: 1.8em;
        font-weight: bold;
        text-align: center;
    }
    .info-box {
        background-color: #E6F3F7; /* Light blue background for info boxes */
        border-left: 5px solid #007BFF; /* Blue border */
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
        st.warning("Please ensure 'model.h5' is in the correct directory.")


# --- Hero Section ---
st.markdown(f'<h1 class="main-header">🥔 Potato Disease Classification</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="tagline">Leveraging Deep Learning for Healthier Potato Crops</p>', unsafe_allow_html=True)
add_vertical_space(2)


# --- Introduction Section ---
st.markdown(f'<h2 class="section-header">Introduction</h2>', unsafe_allow_html=True)
st.markdown("""
In the agriculture industry, farmers often face challenges in identifying diseases in potato plants, such as early blight, late blight, or determining if the plant is healthy. This uncertainty makes it difficult for farmers to apply the appropriate fertilizers and treatments, impacting crop yield and quality. By leveraging machine learning technology, our solution aims to improve agricultural practices, optimize resource allocation, and ultimately enhance the production of healthy potato plants.

This web application, powered by a Convolutional Neural Network (CNN) deep learning model, provides real-time classification of potato plant diseases (healthy, late blight, early blight), empowering farmers with early detection for intervention and prevention.
""")
add_vertical_space(2)

# --- Technologies Section ---
st.markdown(f'<h2 class="section-header">Key Technologies & Skills</h2>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🌐 Python")
    st.markdown("🧠 TensorFlow")
    st.markdown("🖼️ Keras")
with col2:
    st.markdown("🔢 NumPy")
    st.markdown("📊 Matplotlib")
    st.markdown("🚀 Streamlit")
with col3:
    st.markdown("💡 CNN (Convolutional Neural Network)")
    st.markdown("➕ Streamlit-Extras")
add_vertical_space(2)

# --- Installation Section ---
st.markdown(f'<h2 class="section-header">Installation</h2>', unsafe_allow_html=True)
with st.expander("Show Installation Steps"):
    st.markdown("""
    To run this project, you need to install the following packages. It's recommended to use a `requirements.txt` file.

    1.  **Create `requirements.txt`:**
        ```
        numpy
        Pillow
        tensorflow
        streamlit
        streamlit-extras
        ```
    2.  **Install packages:**
        ```bash
        pip install -r requirements.txt
        ```
    """)
add_vertical_space(2)

# --- Usage Section ---
st.markdown(f'<h2 class="section-header">Usage</h2>', unsafe_allow_html=True)
with st.expander("Show Usage Instructions"):
    st.markdown("""
    To use this project, follow these steps:

    1.  **Ensure `model.h5` is in the same directory** as your `app.py` script.
    2.  **Run the Streamlit app:**
        ```bash
        streamlit run app.py
        ```
    3.  **Access the app** in your browser at `http://localhost:8501` (or the address shown in your terminal).
    """)
add_vertical_space(2)

# --- Features Section ---
st.markdown(f'<h2 class="section-header">Features</h2>', unsafe_allow_html=True)

st.markdown("""
Our solution covers the entire machine learning pipeline, from data acquisition to deployment:
""")

st.subheader("1. Data Collection")
st.markdown("""
We obtained the potato disease image dataset from Kaggle, a renowned platform for datasets and data science resources. This dataset consists of images depicting diseased potato plant leaves, meticulously labeled into categories such as early blight, healthy, and late blight.

This collection serves as a valuable asset for training and evaluating our deep learning model, facilitating the development of an effective solution for potato disease classification.
📙 **Dataset Link:** [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
""")
# st.image("images/dataset_example.png", caption="Example images from the dataset", width=600) # Uncomment if you have an image

st.subheader("2. Preprocessing")
st.markdown("""
This phase prepares the raw image data for model training:
""")
st.markdown("- **Image Reading and Resizing:** We leverage TensorFlow to read all images from the directory. Each image undergoes resizing to a standardized dimension of **256x256 pixels**. Furthermore, we organize the processed images into batches with a size of **32**, thus forming a structured dataset ready for subsequent analysis.")
st.markdown("- **Dataset Splitting:** To facilitate comprehensive model evaluation, we partition the dataset into three distinct subsets: **training, validation, and testing**.")
st.markdown("- **Data Pipeline Optimization:** We optimize the data pipeline using TensorFlow's built-in functionalities. The `cache` function is employed to circumvent repetitive loading and reading of training images, and the `prefetch` function enhances training speed by proactively preparing subsequent batches.")

st.subheader("3. Model Building and Training")
st.markdown("""
This section details the core of our deep learning solution:
""")
st.markdown("- **Model Building:** We construct the model architecture using Keras, incorporating layers for resizing, rescaling, random flip, and random rotation for robust preprocessing. A **Convolutional Neural Network (CNN)** architecture is implemented, comprising convolutional layers, pooling layers, and dense layers with adjustable filters/units and activation functions.")
st.markdown("- **Training:** We utilize the **Adam optimizer**, **sparse_categorical_crossentropy loss function**, and **Accuracy metrics**. The model was trained on a **9,000-image dataset** (70/30 train-test split).")

st.markdown("""
    **Model Performance Improvements:**
    * **Improved Accuracy:** From **66.8% to 86.9% in Stage 1** and further to **89.2% in Stage 2**, achieving a **33% overall increase**.
    * **Optimized Training:** Reduced epochs from **9 to 5**, preventing overfitting and cutting training time from **20 minutes to 11 minutes** (a 45% reduction).
""")
# st.image("images/model_architecture.png", caption="Simplified CNN Architecture", width=600) # Uncomment if you have an image

st.subheader("4. Model Deployment and Inference")
st.markdown("""
Following the completion of model training and evaluation, the trained model is saved to enable seamless deployment and inference on new images for classification purposes. This user-friendly Streamlit application allows users to upload new images and obtain real-time classification results.
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
st.markdown(f'<p style="text-align: center; color: #888888; font-size: 0.9em;">Developed with ❤️ using Streamlit and TensorFlow.</p>', unsafe_allow_html=True)

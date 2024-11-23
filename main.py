import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile

st.markdown(
    """
    <style>
    body {
        background-color: #2b2b2b;
        font-family: Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #37474f; 
        color: #ffffff;  
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h2 {
        color: #8bc34a;  
    }
    .sidebar .sidebar-content .stRadio > label {
        font-size: 18px;  
        color: #ffffff;   
    }
    h1, h2, h3 {
        color: #8bc34a;  
        font-weight: bold;
        text-shadow: 1px 1px 2px black;  
    }
    .stImage > img {
        border: 2px solid #8bc34a;  
    }
    .stButton>button {
        background-color: #8bc34a;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 2px 2px 5px gray;
    }
    .stButton>button:hover {
        background-color: #7cb342;
    }
    .block-container {
        padding: 1rem 2rem;  
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function for making predictions
def model_prediction(test_image_path):
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) / 255.0  # Normalize the image
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# class names
class_name = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry (including sour) - Powdery Mildew",
    "Cherry (including sour) - Healthy",
    "Corn (maize) - Cercospora Leaf Spot (Gray Leaf Spot)",
    "Corn (maize) - Common Rust",
    "Corn (maize) - Northern Leaf Blight",
    "Corn (maize) - Healthy",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy",
    "Orange - Huanglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Bell Pepper - Bacterial Spot",
    "Bell Pepper - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy"
]

# Sidebar navigation
st.sidebar.title("🌿 Dashboard")
app_mode = st.sidebar.radio("Select a Page", ["🏠 Home", "ℹ️ About", "🔬 Disease Recognition"])

# Home page
if app_mode == "🏠 Home":
    st.title("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.webp", use_container_width=True, caption="Helping Farmers Diagnose Plant Diseases")
    st.markdown(
        """
        Welcome to the **Plant Disease Recognition System**! 
        This platform helps you identify plant diseases from images and suggests better care for your crops. 🌟
        
        **Features:**
        - 🖼 **Upload an image** of a plant leaf.
        - 🔍 **Detect disease type** in seconds.
        - 🌟 **Easy-to-use** and accurate predictions.

        Let's help farmers improve productivity and fight plant diseases! 🚜
        """
    )

# About page
elif app_mode == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        ### 📂 Dataset Overview
        This dataset includes **87,000+ high-resolution images** of crop leaves, categorized into 38 classes of healthy and diseased conditions.

        **Goal:** 
        To help farmers, gardeners, and agronomists detect plant diseases early, ensuring timely intervention.

        **Technologies Used:**
        - 💻 **TensorFlow** for deep learning.
        - 🌟 **Streamlit** for an interactive UI.
        - 📊 Preprocessed data for faster predictions.

        🌍 Join the mission to combat global agricultural challenges! 
        """
    )
    st.image("about_page.webp", use_container_width=True, caption="Healthy Crops, Healthy Planet! 🌾")

# Disease recognition page
elif app_mode == "🔬 Disease Recognition":
    st.title("🔬 Disease Recognition")
    test_image = st.file_uploader("📤 Upload an Image of a Leaf:", type=["jpg", "png", "jpeg"])
    
    if test_image is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(test_image.read())
            tmp_file_path = tmp_file.name

        st.image(test_image, use_container_width=True, caption="Uploaded Image")

        if st.button("🔍 Predict Disease"):
            status_placeholder = st.empty()
            status_placeholder.text("Analyzing the image... ⏳")

            result_index = model_prediction(tmp_file_path)

            status_placeholder.empty()

            if result_index is not None:
                st.success(f"🌟 Model Prediction: **{class_name[result_index]}**")

        st.info("🤔 Ensure the leaf is clearly visible in the uploaded image!")

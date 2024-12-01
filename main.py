import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    return np.argmax(predictions) 

st.sidebar.title("🌿 Plant Disease Recognition")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 Home", "ℹ️ About", "🔬 Disease Recognition"])

if app_mode == "🏠 Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.webp", use_container_width=True) 
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest! 🌾

    ### How It Works:
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

elif app_mode == "ℹ️ About":
    st.header("ℹ️ About This Project")
    st.markdown("""
    #### About Dataset
    This dataset consists of **87,000+ high-resolution images** of crop leaves, categorized into 38 classes of healthy and diseased conditions.

    #### Content:
    - **train:** 70,295 images
    - **test:** 33 images
    - **validation:** 17,572 images

    **Goal:** To help farmers and gardeners identify plant diseases early for timely intervention.

    **Technologies Used:**
    - **TensorFlow** for deep learning
    - **Streamlit** for interactive UI
    """)

elif app_mode == "🔬 Disease Recognition":
    st.header("🔬 Disease Recognition")

    test_image = st.file_uploader("Choose an Image of a Leaf to Diagnose", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        st.image(test_image, width=400, use_container_width=True, caption="Uploaded Image")  

    if st.button("🔍 Predict Disease"):
        if test_image is not None:
            with st.spinner('Analyzing the image... ⏳'):
                result_index = model_prediction(test_image)
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot', 
                    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 
                    'Grape___Esca', 'Grape___Leaf_blight', 'Grape___healthy', 'Orange___Citrus_greening', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight', 
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 
                    'Tomato___Target_Spot', 'Tomato___Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]
                st.success(f"🌿 **Prediction:** The plant disease is **{class_name[result_index]}**.")
        else:
            st.warning("Please upload an image to proceed with the prediction.")
    
    st.info("Make sure the leaf is clear and well-lit for better prediction accuracy. 📸")

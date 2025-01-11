import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import io

# Load the model
#MODEL = tf.keras.models.load_model("/Users/manojrammopati/Project/DATA SCIENCE ML PROJECTS/Potato Disease Classifier/Model/1.keras")
MODEL = tf.keras.models.load_model("Model/1.keras")
CLASS_NAMES = ["EARLY BLIGHT", "LATE BLIGHT", "HEALTHY"]

# Function to read file as image
def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data))
        image = image.resize((256, 256))  # Resize the image to the expected input shape
        return np.array(image)
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image.")
        return None

# Streamlit frontend
st.set_page_config(page_title="Potato Disease Classifier", page_icon=":potato:")

st.title("Potato Disease Classifier")
st.write("Upload an image of a potato leaf to classify its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        img_bytes = uploaded_file.read()
        
        # Convert the image to a PIL Image
        image = read_file_as_image(img_bytes)
        
        if image is not None:
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            img_batch = np.expand_dims(image, 0)

            # Make prediction
            prediction = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
            confidence = np.max(prediction[0]) * 100

            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image.")
else:
    st.info("Please upload an image file to get started.")

# Add some footer information
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>Potato Disease Classifier - Helping farmers identify potato diseases quickly and accurately.</p>
    </div>
    """,
    unsafe_allow_html=True
)
# filepath: /Users/manojrammopati/Project/DATA SCIENCE ML PROJECTS/Potato Disease Classifier/app.py
import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from io import BytesIO
import uvicorn
import threading

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL = tf.keras.models.load_model("/Users/manojrammopati/Project/DATA SCIENCE ML PROJECTS/Potato Disease Classifier/Model/1.keras")
CLASS_NAMES = ["EARLY BLIGHT", "LATE BLIGHT", "HEALTHY"]

@app.get("/ping")
async def ping():
    return "server is live"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))  # Resize the image to the expected input shape
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return JSONResponse({"prediction": predicted_class, "confidence": float(confidence)})

# Run FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(app, host='localhost', port=8000)

threading.Thread(target=run_fastapi, daemon=True).start()

# Streamlit frontend
st.title("Potato Disease Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert the image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    # Send the image to the FastAPI backend
    response = requests.post("http://localhost:8000/predict", files={"file": img_bytes})

    if response.status_code == 200:
        result = response.json()
        st.write(f"Prediction: {result['prediction']}")
        st.write(f"Confidence: {result['confidence']}")
    else:
        st.write("Error: Unable to get prediction.")
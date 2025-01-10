from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

MODEL = tf.keras.models.load_model("/Users/manojrammopati/Project/DATA SCIENCE ML PROJECTS/Potato Disease Classifier/Model/1.keras")

CLASS_NAMES = ["EARLY BLIGHT","LATE BLIGHT","HEALTHY"]

@app.get("/ping")
async def ping():
    return "server is live"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))  # Resize the image to the expected input shape
    return np.array(image)
   # return np.array(Image.open(BytesIO(data)))


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
   # MODEL.predict(image)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {"prediction": predicted_class , "confidence": float(confidence)}


#app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("/Users/manojrammopati/Project/DATA SCIENCE ML PROJECTS/Potato Disease Classifier/api/index.html") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
    


    

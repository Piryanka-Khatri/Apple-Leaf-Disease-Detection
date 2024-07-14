from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL_PATH = "../AppleLeaf Disease Detection/Apple_Leaf_Disease_detection.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

CLASS_NAMES = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']


def read_file_as_an_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_an_image(await file.read())
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    predictions = MODEL.predict(image)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, port=8000)

from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import os
import gdown

from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Get Google Drive File ID from environment variables
FILE_ID = os.getenv("GDRIVE_FILE_ID")

if not FILE_ID:
    raise ValueError("FILE_ID is not set. Check your .env file.")
# ✅ Google Drive Model File ID (Replace with your actual ID)

MODEL_PATH = "dogs_vs_cats_model.keras"

# ✅ Construct Correct Google Drive URL
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ✅ Download the model correctly using `gdown --fuzzy`
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True, quiet=False)

# ✅ Check if the model downloaded correctly
if os.path.getsize(MODEL_PATH) < 10_000_000:  # Less than 10MB
    raise RuntimeError("Model download failed! File size is too small.")

# ✅ Load the model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Function to preprocess image
def preprocess_image(file):
    try:
        img = Image.open(BytesIO(file)).convert("RGB")  # Convert to RGB
        img = img.resize((256, 256))  # Resize to match model input
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

# ✅ API Endpoint for Prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()  # Read uploaded image
        img_array = preprocess_image(contents)  # Preprocess image
        
        # Get prediction
        prediction = model.predict(img_array)[0][0]  
        label = "Dog" if prediction > 0.5 else "Cat"
        confidence = round(float(prediction) if label == "Dog" else 1 - float(prediction), 2)

        return {"class": label, "confidence": confidence}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def home():
    return {"message": "Welcome to the Image Classification API! Use /predict to classify images."}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

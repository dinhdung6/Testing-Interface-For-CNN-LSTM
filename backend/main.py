from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models (Make sure these files exist!)
models = {
    "LSTM": tf.keras.models.load_model("lstm.keras"),
    "CNN": tf.keras.models.load_model("cnn.h5"),
    "CNN+LSTM": tf.keras.models.load_model("cnn_lstm.h5"),
    
}

scaler = StandardScaler
@app.post("/predict")
async def predict(model_name: str, input_data: list):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model name")

    model = models[model_name]
    input_array = np.array(input_data).reshape(1, -1)  # Adjust shape if needed
    if model_name == "LSTM" or model_name == "CNN":
        if input_array.shape != (20,80):
            raise HTTPException(status_code=400, detail="Input shape must be (20,80)")
        input_array = input_array.reshape(1,20,80)

    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)

    prediction = model.predict(input_tensor)
    result = "Damaged" if prediction[0][0]>= 0.5 else "Undamged"
    
    return {"model_used": model_name, "prediction": prediction}

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...), model_name: str = Form(...)):

    try:
        # Ensure model exists
        if model_name not in models:
            raise ValueError(f"🚨 Invalid model name: {model_name}. Choose from {list(models.keys())}")

        model = models[model_name]  # Select the model

        # Load CSV data
        df = pd.read_csv(file.file)
        print(f"📂 Received file: {file.filename}, Shape: {df.shape}")

        # ✅ Handle extra columns (truncate last 10 columns if needed)
        if df.shape[1] == 90:
            print("🔄 Truncating last 10 columns to match 80 features.")
            df = df.iloc[:, :-10]  # Keep first 80 columns

        # Validate correct input shape
        if df.shape[1] != 80:
            raise ValueError(f"🚨 Incorrect number of columns! Expected 80, but got {df.shape[1]}")
        if len(df) < 20:
            raise ValueError(f"🚨 Not enough time steps! Expected at least 20 rows, but got {len(df)}")

        # Reshape for model input
        input_data = df.values[:20].reshape(1, 20, 80)  # (1 batch, 10 timesteps, 80 features)
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Make prediction
        prediction = model.predict(input_tensor)
        result = "Damaged" if prediction[0][0] >= 0.5 else "Undamaged"

        return {"model_used": model_name, "prediction": result}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
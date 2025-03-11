from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

class TextInput(BaseModel):
    text: str

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
SVD_PATH = "models/svd.pkl"

model, vectorizer, svd = None, None, None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
else:
    logger.error("Model file not found.")

if os.path.exists(VECTORIZER_PATH):
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load the vectorizer: {e}")
else:
    logger.error("Vectorizer file not found.")

if os.path.exists(SVD_PATH):
    try:
        svd = joblib.load(SVD_PATH)
        logger.info("SVD loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load the SVD: {e}")
else:
    logger.error("SVD file not found.")

@app.post("/predict/")
async def predict_sentiment(input_data: TextInput):
    if model is None or vectorizer is None or svd is None:
        raise HTTPException(status_code=500, detail="Model, vectorizer, or SVD is not available.")

    text = input_data.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        text = text.lower()
        logger.info(f"Received input: {text}")

        text_vectorized = vectorizer.transform([text])
        text_svd = svd.transform(text_vectorized)

        prediction = model.predict(text_svd)
        sentiment = int(prediction[0])
        ans = "Positive" if sentiment == 1 else "Negative"
        logger.info(f"Prediction: {sentiment}")
        return {"sentiment": ans}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error in prediction.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

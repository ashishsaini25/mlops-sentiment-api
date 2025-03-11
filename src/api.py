from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import os
from loguru import logger

# Configure Loguru
logger.add("logs/api_log_{time}.log", rotation="1 MB", retention="10 days")

app = FastAPI()

class TextInput(BaseModel):
    text: str

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
SVD_PATH = "models/svd.pkl"

model, vectorizer, svd = None, None, None

def load_artifact(path, name):
    if os.path.exists(path):
        try:
            artifact = joblib.load(path)
            logger.info(f"{name} loaded successfully.")
            return artifact
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
    else:
        logger.error(f"{name} file not found.")
    return None

model = load_artifact(MODEL_PATH, "Model")
vectorizer = load_artifact(VECTORIZER_PATH, "Vectorizer")
svd = load_artifact(SVD_PATH, "SVD")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

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
        sentiment = "Positive" if int(prediction[0]) == 1 else "Negative"

        logger.info(f"Prediction: {sentiment}")
        return {"sentiment": sentiment}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error in prediction.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

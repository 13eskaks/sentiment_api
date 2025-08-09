import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from datetime import datetime

from app.model.models import SentimentModel
from .config.filter import request_id_filter
from .schemas import TextInput, SentimentResponse
from .utils import polarity_to_stars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = SentimentModel()


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # Generates a unique request_id for each request
    request_id_filter.request_id = str(uuid.uuid4())
    response = await call_next(request)
    return response


@app.post("/sentiment/", response_model=SentimentResponse)
async def analyze_sentiment(input: TextInput):
    logger.info("📥 Received request")
    texto = input.texto.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="Empty text not allowed")

    logger.info("🚀 Running sentiment model...")
    sentiment_scores = model.predict(texto)
    positivity = sentiment_scores.get("positividad", 0.0)
    negativity = sentiment_scores.get("negatividad", 0.0)
    neutrality = sentiment_scores.get("neutralidad", 0.0)
    rating = polarity_to_stars(positivity, negativity, neutrality)
    timestamp = datetime.utcnow().isoformat()
    logger.info("✅ Model returned: {sentiment_scores}")

    return SentimentResponse(
        texto=texto,
        positividad=positivity,
        negatividad=negativity,
        neutralidad=neutrality,
        rating=rating,
        fecha=timestamp,
        length=len(texto)
    )

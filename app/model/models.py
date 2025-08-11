import logging
import os

import torch
import torch.nn.functional as F
from app.config.config import HUGGINGFACE_REPO
from app.model.modeling_sentiment import BertForSentimentRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import BertConfig, BertTokenizerFast


class SentimentModel:
    def __init__(self):
        HF_TOKEN = os.getenv("HF_TOKEN")

        logger.info("Starting to load tokenizer...")
        self.tokenizer = BertTokenizerFast.from_pretrained(
            HUGGINGFACE_REPO,
            token=HF_TOKEN
        )
        logger.info("Tokenizer loaded successfully.")

        logger.info("Starting to load model configuration...")
        config = BertConfig.from_pretrained(
            HUGGINGFACE_REPO,
            token=HF_TOKEN
        )
        logger.info("Model configuration loaded successfully.")

        logger.info("Starting to load the model...")
        self.model = BertForSentimentRegression.from_pretrained(
            HUGGINGFACE_REPO,
            config=config,
            token=HF_TOKEN
        )
        self.model.eval()
        logger.info("Model loaded and ready for evaluation.")

        self.labels = ["positividad", "negatividad", "neutralidad"]

    def predict(self, text: str):
        logger.info("🔄 Tokenizing input...")
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        logger.info("🧠 Passing through model...")
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if not isinstance(logits, torch.Tensor):
                logits = logits.logits  # por si hay envoltorio
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        logger.info(f"📈 Softmax probabilities: {probs}")
        sentiment_scores = {label: prob for label, prob in zip(self.labels, probs)}
        return sentiment_scores

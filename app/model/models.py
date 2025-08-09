import logging
import torch
import torch.nn.functional as F
from app.config.config import HUGGINGFACE_REPO
from app.model.modeling_sentiment import BertForSentimentRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import BertConfig, BertTokenizerFast


class SentimentModel:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(HUGGINGFACE_REPO)

        config = BertConfig.from_pretrained(HUGGINGFACE_REPO)
        self.model = BertForSentimentRegression.from_pretrained(HUGGINGFACE_REPO, config=config)
        self.model.eval()

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

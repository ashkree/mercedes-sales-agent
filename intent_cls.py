# app/intent_cls.py

from transformers import pipeline
from typing import Dict

class IntentSentimentClassifier:
    def __init__(self):
        # Load models once on startup
        self.intent_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")  # or your own fine-tuned model
        self.sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def classify_intent(self, text: str) -> Dict:
        """Returns the intent class and confidence."""
        result = self.intent_model(text)[0]
        return {
            "type": "intent",
            "label": result["label"],
            "score": result["score"]
        }

    def classify_sentiment(self, text: str) -> Dict:
        """Returns the sentiment (POSITIVE / NEGATIVE) and confidence."""
        result = self.sentiment_model(text)[0]
        return {
            "type": "sentiment",
            "label": result["label"],
            "score": result["score"]
        }

    def classify_all(self, text: str) -> Dict:
        """Returns both intent and sentiment for a given utterance."""
        return {
            "intent": self.classify_intent(text),
            "sentiment": self.classify_sentiment(text)
        }

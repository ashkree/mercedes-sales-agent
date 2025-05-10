# app/intent_cls.py
# app/intent_cls.py

from transformers import pipeline
from typing import Dict

class IntentSentimentClassifier:
    def __init__(self):
        # Retain fast sentiment classifier
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def classify_intent(self, text: str) -> Dict:
        """
        Classifies intent using a local LLaMA-based model.
        Categories: informed, exploratory, unknown.
        """
        prompt = (
            "You are an AI assistant for Mercedes-Benz.\n"
            "Based on the user's message, classify their intent as:\n"
            "- informed: they know exactly what they want\n"
            "- exploratory: they need help or are unsure\n"
            "- unknown: not enough context\n\n"
            f'User: "{text}"\n'
            "Intent:"
        )

        label = self.llm_infer(prompt).strip().lower()

        if label not in {"informed", "exploratory", "unknown"}:
            label = "unknown"

        return {
            "type": "intent",
            "label": label,
            "confidence": 1.0  # assumed since LLaMA has no explicit score
        }

    def classify_sentiment(self, text: str) -> Dict:
        """
        Returns sentiment (POSITIVE / NEGATIVE) and confidence.
        """
        result = self.sentiment_model(text)[0]
        return {
            "type": "sentiment",
            "label": result["label"],
            "score": result["score"]
        }

    def classify_all(self, text: str) -> Dict:
        """
        Returns both intent and sentiment for a given user message.
        """
        return {
            "intent": self.classify_intent(text),
            "sentiment": self.classify_sentiment(text)
        }

    def llm_infer(self, prompt: str) -> str:
        """
        Call your local LLaMA chat model with the given prompt.
        Replace this stub with actual integration.
        """
        print("🧠 Prompt to LLaMA:\n", prompt)
        # Example: return llama_cpp_infer(prompt)
        return "exploratory"  # placeholder


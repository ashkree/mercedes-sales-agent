# app/intent_cls.py

from transformers import pipeline
from typing import Dict


class IntentSentimentClassifier:
    def __init__(self):
        # Zero-shot intent classifier
        self.intent_model = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

        # Sentiment classifier
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        # Hypotheses aligned to customer-facing assistant role
        self.hypotheses = {
            "informed": "The customer knows exactly what kind of car they want.",
            "exploratory": "The customer is still deciding and needs help exploring options.",
            "test_drive": "The customer wants to book a test drive.",
            "compare_models": "The customer is comparing two or more car models.",
            "price_inquiry": "The customer is asking about car prices, offers, or financing.",
            "availability": "The customer wants to know if a specific car is available.",
            "booking": "The customer is trying to schedule a showroom visit or callback.",
            "after_sales": "The customer is asking about service, warranty, or support.",
            "exit": "The customer wants to end the conversation."
        }

        # Optional keyword-based intent reinforcement
        self.intent_keywords = {
            "compare_models": ["compare", "difference", "vs", "versus"],
            "test_drive": ["test drive", "book drive", "schedule drive"],
            "price_inquiry": ["price", "cost", "how much", "financing", "payment"],
            "availability": ["available", "in stock", "delivery", "waitlist"],
            "booking": ["appointment", "schedule", "callback", "visit"],
            "after_sales": ["service", "warranty", "maintenance"],
            "exit": ["bye", "goodbye", "end", "leave"]
        }

    def classify_intent(self, text: str) -> Dict:
        """
        Classify customer intent using zero-shot classification and keyword cues.
        """
        # Handle empty input
        if not text or text.strip() == "":
            return {
                "type": "intent",
                "label": "exit",
                "confidence": 1.0,
                "all_scores": {"exit": 1.0}
            }

        result = self.intent_model(text, list(self.hypotheses.values()))

        label_scores = {
            key: float(score)
            for key, hyp in self.hypotheses.items()
            for label, score in zip(result["labels"], result["scores"])
            if label == hyp
        }

        # Keyword reinforcement (soft boost)
        text_lower = text.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(k in text_lower for k in keywords):
                label_scores[intent] = label_scores.get(intent, 0) + 0.2

        # Determine top intent
        top_label = max(label_scores, key=label_scores.get)
        top_conf = label_scores[top_label]

        return {
            "type": "intent",
            "label": top_label,
            "confidence": top_conf,
            "all_scores": label_scores
        }

    def classify_sentiment(self, text: str) -> Dict:
        """
        Classify sentiment as POSITIVE or NEGATIVE.
        """
        # Handle empty input
        if not text or text.strip() == "":
            return {
                "type": "sentiment",
                "label": "NEUTRAL",
                "score": 0.5
            }

        result = self.sentiment_model(text)[0]
        return {
            "type": "sentiment",
            "label": result["label"],
            "score": float(result["score"])
        }

    def classify_all(self, text: str) -> Dict:
        """
        Return both intent and sentiment classifications.
        """
        return {
            "intent": self.classify_intent(text),
            "sentiment": self.classify_sentiment(text)
        }


# CLI for testing
if __name__ == "__main__":
    clf = IntentSentimentClassifier()
    print("💬 Type a customer message (or 'exit' to quit):\n")

    while True:
        query = input("🧑 You: ")
        if query.lower() in {"exit", "quit"}:
            break

        result = clf.classify_all(query)

        print("\n🤖 Result:")
        print(
            f"Top Intent: {result['intent']['label']} ({result['intent']['confidence']:.2f})")
        print("All intent scores:")
        for label, score in sorted(result['intent']['all_scores'].items(), key=lambda x: -x[1]):
            print(f"  - {label:15}: {score:.2f}")
        print(
            f"Sentiment: {result['sentiment']['label']} ({result['sentiment']['score']:.2f})\n")

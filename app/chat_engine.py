# app/chat_engine.py

from app.intent_cls import IntentSentimentClassifier
from app.query_interface import search
from app.response_handler import generate_llama_response

intent_clf = IntentSentimentClassifier()


def generate_prompt(user_input: str, intent: str, sentiment: str, chunks: list[str]) -> str:
    context = "\n".join(f"- {c}" for c in chunks) if chunks else "None"

    return f"""
        You are a helpful AI assistant for Mercedes-Benz. Your job is to guide car buyers in a friendly and expert way.

        User intent: {intent}
        User sentiment: {sentiment}

        User message: "{user_input}"

        Relevant information from our car database:
        {context}

        Respond in a helpful, conversational tone as if you are guiding a real customer.
        Assistant:"""


def handle_user_input(user_input: str) -> str:
    analysis = intent_clf.classify_all(user_input)
    intent = analysis["intent"]["label"]
    sentiment = analysis["sentiment"]["label"]

    # Only retrieve chunks if intent is "informed" or "price_inquiry"
    if intent in {"informed", "price_inquiry", "compare_models"}:
        chunks = search(user_input)
    else:
        chunks = []

    prompt = generate_prompt(user_input, intent, sentiment, chunks)
    return generate_llama_response(prompt)

# app/chat_engine.py

from app.intent_cls import IntentSentimentClassifier
from app.query_interface import search
from app.llm_backend import generate

intent_clf = IntentSentimentClassifier()


def generate_prompt(user_input: str, intent: str, sentiment: str, chunks: list[str]) -> str:
    context = "\n".join(f"- {c}" for c in chunks) if chunks else "None"

    # Create a more detailed system prompt based on intent
    system_instructions = ""
    if intent == "informed":
        system_instructions = "The customer seems to know what they want. Provide specific information about the models they're asking about."
    elif intent == "exploratory":
        system_instructions = "The customer is exploring options. Help them understand the different models and suggest appropriate ones based on their needs."
    elif intent == "test_drive":
        system_instructions = "The customer is interested in a test drive. Guide them on how to book one and what to expect."
    elif intent == "compare_models":
        system_instructions = "The customer wants to compare models. Provide a clear comparison of the models mentioned."
    elif intent == "price_inquiry":
        system_instructions = "The customer is asking about prices. Be specific about pricing and mention any current offers."
    elif intent == "availability":
        system_instructions = "The customer wants to know about availability. Provide information about stock and delivery times."
    elif intent == "booking":
        system_instructions = "The customer wants to book an appointment. Guide them on how to do so."
    elif intent == "after_sales":
        system_instructions = "The customer is asking about after-sales services. Provide information about warranty, maintenance, etc."
    elif intent == "exit":
        system_instructions = "The customer wants to end the conversation. Say goodbye politely."

    # Phi-2 works well with this prompt format
    return f"""
You are a helpful AI assistant for Mercedes-Benz Gargash, a luxury car dealer in the UAE. Your job is to guide car buyers in a friendly and expert way.

User intent: {intent}
User sentiment: {sentiment}
Additional instructions: {system_instructions}

User message: "{user_input}"

Relevant information from our Mercedes-Benz database:
{context}

Respond in a helpful, conversational tone as if you are guiding a real customer at a Mercedes-Benz dealership in the UAE. 
Be concise but informative, and always maintain the premium luxury feel of the Mercedes-Benz brand.
If the customer is asking about prices, always mention that prices are in AED (UAE Dirhams).
Assistant:"""


def handle_user_input(user_input: str) -> str:
    """
    Process user input and generate a response.

    Args:
        user_input (str): The user's message

    Returns:
        str: The assistant's response
    """
    # Handle empty input gracefully
    if not user_input or user_input.strip() == "":
        return "I'm here to help with any questions about Mercedes-Benz vehicles. What would you like to know?"

    try:
        # Analyze intent and sentiment
        analysis = intent_clf.classify_all(user_input)
        intent = analysis["intent"]["label"]
        sentiment = analysis["sentiment"]["label"]

        # Retrieve relevant information based on intent
        if intent in {"informed", "price_inquiry", "compare_models", "availability"}:
            # For these intents, we want to search for relevant car info
            chunks = search(user_input)
        else:
            chunks = []

        # Generate prompt and get response
        prompt = generate_prompt(user_input, intent, sentiment, chunks)
        return generate(prompt)

    except Exception as e:
        # Provide a graceful fallback in case of errors
        print(f"Error processing input: {e}")
        return "I apologize for the inconvenience. I'm having trouble processing your request. How else can I assist you with our Mercedes-Benz vehicles?"

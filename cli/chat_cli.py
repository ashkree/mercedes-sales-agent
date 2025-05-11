# cli/chat_cli.py

from app.chat_engine import handle_user_input


def main():
    print("🚗 Mercedes-Benz AI Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("🧑 You: ")

        # Check for exit command
        if user_input.lower() in {"exit", "quit", "goodbye", "bye"}:
            print("👋 Thank you for considering Mercedes-Benz. Have a wonderful day!")
            break

        # Handle empty input gracefully
        if not user_input.strip():
            print("\n🤖 Assistant: I'm here to help with any questions about Mercedes-Benz vehicles. What would you like to know?\n")
            continue

        # Process valid input
        try:
            response = handle_user_input(user_input)
            print(f"\n🤖 Assistant: {response}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\n🤖 Assistant: I apologize for the inconvenience. Let me know how I can assist you with our Mercedes-Benz vehicles.\n")


if __name__ == "__main__":
    main()

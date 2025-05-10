# cli/chat_cli.py

from app.chat_engine import handle_user_input


def main():
    print("🚗 Mercedes-Benz AI Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        response = handle_user_input(user_input)
        print(f"\n🤖 Assistant: {response}\n")


if __name__ == "__main__":
    main()

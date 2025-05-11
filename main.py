# run.py

import os
import sys
import time
from contextlib import contextmanager


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr"""
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Open null devices
    null_stdout = open(os.devnull, 'w')
    null_stderr = open(os.devnull, 'w')

    try:
        # Redirect stdout/stderr to null devices
        sys.stdout = null_stdout
        sys.stderr = null_stderr
        yield
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Close null devices
        null_stdout.close()
        null_stderr.close()


def show_loading_animation(duration=5):
    """Show a simple loading animation for the specified duration"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("🚗 Mercedes-Benz AI Assistant")
    print("⏳ Loading model, please wait...")

    animation_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()

    while time.time() - start_time < duration:
        for char in animation_chars:
            sys.stdout.write(f"\r{char} Loading Phi-2 model...")
            sys.stdout.flush()
            time.sleep(0.1)


def main():
    """Launch the Mercedes Sales Assistant with suppressed logs"""
    # Show loading animation
    show_loading_animation()

    # Import and initialize in quiet mode
    with suppress_output():
        from cli.chat_cli import main as chat_main

        # Pre-load the LLM to avoid logs later
        from app.llm_backend import llm

    # Clear screen and launch the actual CLI
    os.system('cls' if os.name == 'nt' else 'clear')

    # Run the chat CLI
    chat_main()


if __name__ == "__main__":
    main()

from pathlib import Path
import os
from llama_cpp import Llama

# Optimize Metal performance
os.environ["GGML_METAL_PATH_OVERRIDE"] = str(
    Path(__file__).resolve().parent.parent / "models")

# Path to the quantized Phi-2 model
MODEL_PATH = Path(__file__).resolve().parent.parent / \
    "models" / "phi-2.Q4_K_M.gguf"

# Initialize the LLM (Phi-2) once on import
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,           # context window size for prompt + response
    # utilize physical CPU cores (reduced from 8 for stability)
    n_threads=4,
    n_gpu_layers=32,      # offload all Transformer layers to Apple Metal
    # batch size for token generation (reduced from 128 for stability)
    n_batch=64
)


def generate(prompt: str,
             max_tokens: int = 256,
             temperature: float = 0.7,
             top_p: float = 0.9) -> str:
    """
    Generate a response using the local Phi-2 model.

    Args:
        prompt (str): The full prompt to send to the model.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature (higher = more creative).
        top_p (float): Cumulative probability for nucleus sampling.

    Returns:
        str: The generated text (stripped of leading/trailing whitespace).
    """
    try:
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # Stop generation at these markers
            stop=["User:", "User message:", "🧑 You:"]
        )
        # The `choices` list contains the outputs; we take the first one
        return result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response. How else can I assist you?"

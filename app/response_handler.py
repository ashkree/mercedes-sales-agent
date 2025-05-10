# app/response_handler.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Choose your model — you can swap this out with a smaller one if needed
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"  # Uses GPU if available, otherwise CPU
)


def generate_llama_response(prompt: str, max_tokens: int = 256) -> str:
    """
    Generate a natural-language response using the LLaMA model from Hugging Face.
    """
    input_ids = tokenizer(
        prompt, return_tensors="pt").input_ids.to(model.device)

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

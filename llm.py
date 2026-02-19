import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/"

# Free HF models — ordered by quality
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",   # Primary — best quality
    "HuggingFaceH4/zephyr-7b-beta",          # Fallback #1 — great for instructions
    "google/flan-t5-xxl",                    # Fallback #2 — lightweight & reliable
]

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def query_model(prompt: str, model: str) -> str:
    """Send prompt to a HF Inference API model and return generated text."""

    # Mistral & Zephyr use chat-style input
    if "mistral" in model.lower() or "zephyr" in model.lower():
        payload = {
            "inputs": f"<s>[INST] {prompt} [/INST]",
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "wait_for_model": True,   # Wait during cold start instead of failing
            }
        }
    else:
        # flan-t5 style
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "wait_for_model": True,
            }
        }

    response = requests.post(
        HF_API_URL + model,
        headers=HEADERS,
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    result = response.json()

    # Parse response format
    if isinstance(result, list) and len(result) > 0:
        item = result[0]
        if "generated_text" in item:
            text = item["generated_text"]
            # Strip the input prompt from output (some models echo it back)
            if "[/INST]" in text:
                text = text.split("[/INST]")[-1].strip()
            return text

    raise ValueError(f"Unexpected response format: {str(result)[:200]}")


def generate_narrative(prompt: str) -> str:
    """
    Generate a hospitality concept description with automatic model fallback.

    Attempt order:
      1. Mistral-7B-Instruct  (primary)
      2. Zephyr-7B-Beta        (fallback #1)
      3. Flan-T5-XXL           (fallback #2)
    """
    instruction = (
        "You are a hospitality concept designer. "
        "Write a detailed hospitality concept description:\n"
        + prompt
    )

    last_error = None
    for model in MODELS:
        try:
            print(f"Trying: {model}...")
            text = query_model(instruction, model)
            print(f"✅ Success with {model}")
            return text

        except Exception as e:
            print(f"⚠️  {model} failed: {e}")
            last_error = e

    return "Text generation temporarily unavailable. Please try again later."


# --- Example usage ---
if __name__ == "__main__":
    result = generate_narrative("A boutique beachfront hotel in Bali with a focus on wellness and local culture.")
    print(result)
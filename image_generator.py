import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
HF_API_URL = "https://api-inference.huggingface.co/models/"

# Free models (all available on HF Inference API)
MODELS = [
    "stabilityai/stable-diffusion-2-1",          # Primary
    "runwayml/stable-diffusion-v1-5",             # Fallback #1
    "CompVis/stable-diffusion-v1-4",              # Fallback #2
]

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def generate_with_hf(prompt: str, model: str) -> bytes:
    """Generate image using HF Inference API."""
    response = requests.post(
        HF_API_URL + model,
        headers=HEADERS,
        json={"inputs": prompt},
        timeout=60
    )
    response.raise_for_status()

    # Check if response is actually an image (not an error JSON)
    if "image" not in response.headers.get("Content-Type", ""):
        raise ValueError(f"Unexpected response: {response.text[:200]}")

    return response.content


def generate_image(prompt: str, output_path: str = "generated_image.png") -> str:
    """
    Generate an image with automatic fallback.

    Attempt order:
      1. Stable Diffusion 2.1  (primary)
      2. Stable Diffusion 1.5  (fallback #1)
      3. Stable Diffusion 1.4  (fallback #2)
    """
    last_error = None

    for model in MODELS:
        try:
            print(f"Trying: {model}...")
            image_bytes = generate_with_hf(prompt, model)

            with open(output_path, "wb") as f:
                f.write(image_bytes)

            print(f"✅ Success with {model}")
            return output_path

        except Exception as e:
            print(f"⚠️  {model} failed: {e}")
            last_error = e

    raise RuntimeError(f"All models failed. Last error: {last_error}")


# --- Example usage ---
if __name__ == "__main__":
    path = generate_image("Astronaut riding a horse on Mars")
    print(f"Image saved to: {path}")
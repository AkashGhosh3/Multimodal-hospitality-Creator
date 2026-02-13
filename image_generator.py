import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def generate_image(prompt):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": prompt}
    )

    image_path = "generated_image.png"
    with open(image_path, "wb") as f:
        f.write(response.content)

    return image_path

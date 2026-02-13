import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/models/google/flan-t5-large"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def generate_narrative(prompt):
    instruction = (
        "You are a hospitality concept designer. "
        "Write a detailed hospitality concept description:\n"
        + prompt
    )

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": instruction},
        timeout=90
    )

    if response.status_code != 200:
        return "Text generation temporarily unavailable (free API)."

    try:
        result = response.json()
    except Exception:
        return "Text generation temporarily unavailable (free API)."

    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]

    return "Text generation temporarily unavailable (free API)."

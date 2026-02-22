import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_description(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ✅ Current working Groq model
        messages=[
            {"role": "system", "content": "You are a luxury hospitality concept designer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )

    return response.choices[0].message.content
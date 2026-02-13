from llm import generate_narrative
from image_generator import generate_image
from vector_db import store_prompt

user_prompt = input("Enter hospitality concept: ")

store_prompt(user_prompt)

text_output = generate_narrative(user_prompt)
image_output = generate_image(user_prompt)

print("\nGenerated Narrative:\n", text_output)
print("\nGenerated Image URL:\n", image_output)

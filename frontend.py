import streamlit as st
from llm import generate_narrative
from image_generator import generate_image
from vector_db import store_prompt

st.title("🏨 Multimodal Hospitality Creator")

user_prompt = st.text_input("Enter Hospitality Concept:")

if st.button("Generate Concept"):

    if user_prompt:
        store_prompt(user_prompt)

        st.write("### Generated Narrative")
        text_output = generate_narrative(user_prompt)
        st.write(text_output)

        st.write("### Generated Image")
        image_path = generate_image(user_prompt)

        if image_path.endswith(".png"):
            st.image(image_path)
        else:
            st.write(image_path)

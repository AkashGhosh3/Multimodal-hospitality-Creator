import streamlit as st
from llm import generate_description
from image_generator import generate_image

st.set_page_config(
    page_title="Multimodal Hospitality Creator",
    layout="wide"
)

st.title("🏨 Multimodal Hospitality Concept Creator")

prompt = st.text_input(
    "Enter your hospitality concept:",
    "Luxury beachfront eco-resort in Bali"
)

if st.button("Generate Concept"):

    # Generate description
    with st.spinner("Generating description..."):
        description = generate_description(prompt)

    # Generate image bytes
    with st.spinner("Generating image..."):
        image_bytes = generate_image(prompt)

    col1, col2 = st.columns(2)

    # Left Column - Text
    with col1:
        st.subheader("Concept Description")
        st.write(description)

    # Right Column - Image
    with col2:
        st.subheader("Visual Representation")

        if image_bytes:
            st.image(image_bytes, width=500)
        else:
            st.error("Image generation failed. Please try again.")
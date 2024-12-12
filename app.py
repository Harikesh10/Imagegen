import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Function to load the model (CPU version)
@st.cache_resource
def load_model():
    # Load the Stable Diffusion Pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cpu")  # Use CPU
    return pipe

# Streamlit UI
def main():
    st.title("Text-to-Image Generator")
    st.write("Enter a description below, and the AI will generate an image for you!")

    # Input from user
    user_prompt = st.text_input("Enter a description of the image (e.g., 'A sunset over mountains'):")

    # Generate button
    if st.button("Generate Image"):
        if not user_prompt.strip():
            st.warning("Please enter a valid description!")
        else:
            with st.spinner("Generating image... Please wait!"):
                # Load the model
                pipe = load_model()

                # Generate image
                image = pipe(user_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

                # Display the image
                st.image(image, caption="Generated Image", use_column_width=True)
                st.success("Image generated successfully!")

if __name__ == "__main__":
    main()

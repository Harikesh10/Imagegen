import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

# Load Stable Diffusion model
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cpu")  # Set to CPU
    return pipe

pipe = load_model()

# App Title
st.title("Cool Text-to-Image Generator")
st.write("Generate stunning images from text prompts with additional features!")

# Sidebar for settings
st.sidebar.header("Settings")
image_width = st.sidebar.slider("Image Width (px)", 256, 1024, 512, step=64)
image_height = st.sidebar.slider("Image Height (px)", 256, 1024, 512, step=64)
download_option = st.sidebar.checkbox("Enable Download")

# Main Input Section
prompt = st.text_input("Enter your text description", placeholder="A fantasy castle on a hill during sunset")
generate_button = st.button("Generate Image")

# Image Generation
if generate_button:
    if prompt.strip():
        with st.spinner("Generating your image..."):
            generated_image = pipe(prompt, height=image_height, width=image_width).images[0]
            st.image(generated_image, caption="Generated Image", use_container_width=True)

            # Save and download option
            if download_option:
                buf = io.BytesIO()
                generated_image.save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    label="Download Image",
                    data=buf,
                    file_name="generated_image.png",
                    mime="image/png",
                )
    else:
        st.error("Please enter a valid text description.")

# Gallery Feature
if "image_history" not in st.session_state:
    st.session_state.image_history = []

if generate_button and prompt.strip():
    st.session_state.image_history.append((prompt, generated_image))

if st.session_state.image_history:
    st.subheader("Image History")
    for i, (hist_prompt, hist_image) in enumerate(st.session_state.image_history):
        with st.expander(f"Prompt: {hist_prompt} (Image {i+1})"):
            st.image(hist_image, use_container_width=True)

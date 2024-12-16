import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageEnhance
import io

# Load Stable Diffusion and Anime models
@st.cache_resource
def load_models(device):
    models = {
        "Standard": StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device),
        "Anime": StableDiffusionPipeline.from_pretrained("hakurei/waifu-diffusion").to(device),
    }
    return models

# Device selection cpu or gpu
st.sidebar.header("Device Options")
device_choice = st.sidebar.radio("Choose the device for processing", ["cpu", "cuda"], index=0)

# Load models based on selected device
models = load_models(device_choice)

# App Title
st.title("Enhanced Text-to-Image Generator")
st.write("Generate stunning images with multiple models, styles, and preferences!")

# Sidebar for settings
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()) + ["Black & White", "Enhance", "Realistic"], index=0)
speed_quality = st.sidebar.radio("Preference", ["Speed", "Quality"], index=0)
image_width = st.sidebar.slider("Image Width (px)", 256, 1024, 512, step=64)
image_height = st.sidebar.slider("Image Height (px)", 256, 1024, 512, step=64)
download_option = st.sidebar.checkbox("Enable Download")

# Style Selection
st.sidebar.header("Style Options")
styles = ["Default", "Cyberpunk", "Anime", "Fantasy", "Realistic", "Black & White"]
selected_style = st.sidebar.selectbox("Choose a style", styles, index=0)

# Shape Selection
st.sidebar.header("Shape Options")
shapes = {"Square": (512, 512), "Portrait": (512, 768), "Landscape": (768, 512)}
selected_shape = st.sidebar.selectbox("Choose Shape", list(shapes.keys()))
image_width, image_height = shapes[selected_shape]

# Main Input Section
prompt = st.text_input("Enter your text description", placeholder="A fantasy castle on a hill during sunset")
generate_button = st.button("Generate Image")

# Image Post-Processing Functions
def apply_black_and_white(image):
    return image.convert("L")

def enhance_image(image):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(2.0)

def make_realistic(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)

# Image Generation
if generate_button:
    if prompt.strip():
        with st.spinner("Generating your image..."):
            pipe = models.get(model_choice, models["Standard"])
            if speed_quality == "Speed":
                pipe.enable_attention_slicing()
            else:
                pipe.disable_attention_slicing()

            # Append style to the prompt
            full_prompt = f"{prompt}, {selected_style} style" if selected_style != "Default" else prompt

            generated_image = pipe(full_prompt, height=image_height, width=image_width).images[0]

            # Apply post-processing options
            if model_choice == "Black & White":
                generated_image = apply_black_and_white(generated_image)
            elif model_choice == "Enhance":
                generated_image = enhance_image(generated_image)
            elif model_choice == "Realistic":
                generated_image = make_realistic(generated_image)

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

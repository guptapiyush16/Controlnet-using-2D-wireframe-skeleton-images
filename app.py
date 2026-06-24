import streamlit as st
import torch
from PIL import Image
import os

# --- Page Config ---
st.set_page_config(
    page_title="ControlNet Wireframe Generator",
    page_icon="🎨",
    layout="wide"
)

# --- App Title & Sidebar ---
st.title("🎨 ControlNet: Wireframe to Real Photo")
st.markdown("""
This application uses a **ControlNet** conditioned on 2D wireframes to generate high-fidelity images.
Upload a wireframe image (or use a sample), provide a text prompt, and generate your photo!
""")

with st.sidebar:
    st.header("Settings")
    model_id = st.text_input("Base Model ID", value="runwayml/stable-diffusion-v1-5")
    custom_weights = st.text_input("Custom ControlNet Weights Path", value="")
    
    st.divider()
    st.markdown("### Rendering Tips")
    st.info("For best results, use white wireframes on a black background.")

# --- UI Components ---
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    
    # Prompt Input
    prompt = st.text_area(
        "Enter your prompt:", 
        placeholder="e.g., a realistic sports car, 8k resolution, highly detailed",
        value="a realistic sports car, 8k resolution, highly detailed"
    )
    
    # Wireframe Input
    input_mode = st.radio("Input Wireframe Mode", ["Upload Image", "Use Sample"])
    
    wireframe_img = None
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload a 2D Wireframe Image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            wireframe_img = Image.open(uploaded_file).convert("RGB")
    else:
        sample_path = "data/processed/8ff7f1f2465347cd8b80c9b206c2781e_wire.png"
        if os.path.exists(sample_path):
            st.write("Using sample wireframe from data pipeline.")
            wireframe_img = Image.open(sample_path).convert("RGB")
        else:
            st.error("Sample wireframe not found. Please run the data pipeline first or upload an image.")

    if wireframe_img:
        st.image(wireframe_img, caption="Input Wireframe Condition", use_container_width=True)

# --- Execution ---
with col2:
    st.header("Output")
    generate_btn = st.button("✨ Generate Real Photo", type="primary")
    
    if generate_btn:
        if not prompt:
            st.warning("Please enter a prompt.")
        elif not wireframe_img:
            st.warning("Please provide a wireframe image.")
        else:
            with st.spinner("Generating image... This may take a moment (Model weights download might occur)."):
                try:
                    # Save temporary wireframe for inference script
                    temp_wireframe_path = "temp_wireframe.png"
                    wireframe_img.save(temp_wireframe_path)
                    
                    # Call inference logic
                    from src.inference import generate_image
                    # Note: In a production app, we would cache the pipeline to avoid reloading
                    output_img = generate_image(
                        prompt=prompt,
                        wireframe_image_path=temp_wireframe_path,
                        model_path=custom_weights if custom_weights else None,
                        base_model=model_id,
                        output_path="output/streamlit_generated.png"
                    )
                    
                    # Display Output
                    st.success("Generation Complete!")
                    st.image(output_img, caption="Generated Photo", use_container_width=True)
                    
                    # Download link
                    with open("output/streamlit_generated.png", "rb") as file:
                        st.download_button(
                            label="📥 Download Photo",
                            data=file,
                            file_name="generated_photo.png",
                            mime="image/png"
                        )
                        
                except Exception as e:
                    st.error(f"Error during generation: {e}")
                    st.info("Ensure you have set up the environment and have enough disk space for model weights.")

# --- Footer ---
st.divider()
st.caption("Developed for ControlNet 2D Wireframe Project | Samsung Prism")

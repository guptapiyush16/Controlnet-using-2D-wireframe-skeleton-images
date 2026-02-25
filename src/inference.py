import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import os

def generate_image(
    prompt, 
    wireframe_image_path, 
    model_path=None, 
    base_model="runwayml/stable-diffusion-v1-5",
    output_path="output/generated_image.png"
):
    """
    Generates an image using a ControlNet conditioned on a wireframe image.
    """
    # 1. Load ControlNet
    # If a custom fine-tuned model path is provided, use it. 
    # Otherwise, fallback to a base controlnet (e.g., Canny) for structural testing.
    if model_path and os.path.exists(model_path):
        print(f"Loading custom ControlNet from {model_path}...")
        controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16)
    else:
        print("Model path not provided or not found. Using default Canny ControlNet for demo/structure.")
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

    # 2. Setup Pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, 
        controlnet=controlnet, 
        torch_dtype=torch.float16
    )

    # 3. Speed up/Optimize inference
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        print("Attempting to enable model CPU offload...")
        pipe.enable_model_cpu_offload() 
    except Exception as e:
        print(f"Failed to enable CPU offload: {e}")
        print("Falling back to standard device placement.")
        if torch.cuda.is_available():
            pipe.to("cuda")
        else:
            pipe.to("cpu")

    # 4. Load and Preprocess Wireframe
    if not os.path.exists(wireframe_image_path):
        raise FileNotFoundError(f"Wireframe image not found at {wireframe_image_path}")
        
    wireframe = Image.open(wireframe_image_path).convert("RGB")
    wireframe = wireframe.resize((512, 512))

    # 5. Generate
    print(f"Generating image for prompt: '{prompt}'...")
    generator = torch.manual_seed(42) # For reproducibility
    image = pipe(
        prompt, 
        image=wireframe, 
        num_inference_steps=20, 
        generator=generator
    ).images[0]

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return image

if __name__ == "__main__":
    # User requested demo
    target_wireframe = "temp_wireframe.png"
    if os.path.exists(target_wireframe):
        generate_image(
            prompt="a very sexy red car, hot wheels, high quality, 8k",
            wireframe_image_path=target_wireframe,
            output_path="output/user_demo_generation.png"
        )
    else:
        print(f"Wireframe not found at {target_wireframe}.")

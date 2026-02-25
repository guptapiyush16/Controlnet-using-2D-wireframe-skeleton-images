import torch
import torch.nn.functional as F
from diffusers import (
    ControlNetModel, 
    StableDiffusionControlNetPipeline, 
    DDPMScheduler, 
    AutoencoderKL, 
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import os
from data.dataset import ControlNetDataset
from torch.utils.data import DataLoader

# Training Configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_CONDITIONING_SCALE = 1.0
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
NUM_EPOCHS = 10

def train():
    accelerator = Accelerator(gradient_accumulation_steps=1)
    
    # 1. Load Pre-trained Components
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    
    # 2. Initialize ControlNet
    # We initialize it from the UNet's structure
    controlnet = ControlNetModel.from_unet(unet)
    
    # 3. Freeze other components
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE)
    
    # 5. Dataset & Dataloader
    dataset = ControlNetDataset(data_dir="data/processed")
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 6. Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    
    # 7. Prepare with Accelerator
    controlnet, optimizer, train_dataloader = accelerator.prepare(
        controlnet, optimizer, train_dataloader
    )
    
    # GPU casting
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)
    
    # 8. Training Loop
    for epoch in range(NUM_EPOCHS):
        controlnet.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                
                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                # Simplification: assume static prompt for now
                inputs = tokenizer(batch["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer.model_max_length).to(accelerator.device)
                encoder_hidden_states = text_encoder(inputs.input_ids)[0]
                
                # ControlNet prediction
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=batch["conditioning_pixel_values"],
                    return_dict=False,
                )
                
                # Predict noise with UNet and ControlNet residuals
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # Loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
        print(f"Epoch {epoch} complete. Loss: {loss.item()}")
        # Save checkpoint
        if accelerator.is_main_process:
            controlnet.save_pretrained(f"checkpoints/epoch_{epoch}")

if __name__ == "__main__":
    train()

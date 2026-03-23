import os
# Prevent JAX from pre-allocating all memory.
# Without this, JAX might consume 90% of VRAM/HBM immediately, causing OOM when loading models.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools
from flax.training import train_state
from flax import jax_utils
from diffusers import FlaxUNet2DConditionModel, FlaxDDPMScheduler, FlaxAutoencoderKL
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from PIL import Image, ImageDraw
import argparse

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
class TrainConfig:
    # Set to "runwayml/stable-diffusion-v1-5" for actual training.
    # Using a tiny model by default to ensure it runs instantly on Kaggle without OOM or long downloads.
    pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
    
    # On Kaggle TPU v5e-8, there are 8 cores. Batch size must be divisible by device count.
    batch_size = 8 
    learning_rate = 1e-4
    num_train_steps = 100
    seed = 42
    # Set to 24 to be a multiple of our 8 TPU cores (batch_size=8). 
    num_synthetic_samples = 24 

config = TrainConfig()

# ==========================================
# 2. Generate Synthetic Finetuning Data
# ==========================================
def generate_synthetic_dataset(num_samples):
    """
    Generates a dataset of synthetic PIL images and corresponding text captions.
    This simulates a real dataset before it gets processed by VAE and CLIP.
    We create diverse shapes and colors to simulate real concept binding.
    """
    print(f"Generating {num_samples} synthetic image-text pairs...")
    dataset = []
    
    # Diverse attributes for our toy dataset
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    shapes = ["circle", "rectangle", "triangle"]
    
    for i in range(num_samples):
        # Create a blank white image
        img = Image.new("RGB", (512, 512), color="white")
        draw = ImageDraw.Draw(img)
        
        # Pick a combination of color and shape
        color_name = colors[i % len(colors)]
        shape_name = shapes[(i // len(colors)) % len(shapes)]
        
        # Draw the shape and create a matching caption
        if shape_name == "circle":
            draw.ellipse([128, 128, 384, 384], fill=color_name)
            caption = f"A {color_name} circle on a white background"
        elif shape_name == "rectangle":
            draw.rectangle([100, 150, 412, 362], fill=color_name)
            caption = f"A {color_name} rectangle on a white background"
        elif shape_name == "triangle":
            draw.polygon([(256, 100), (100, 412), (412, 412)], fill=color_name)
            caption = f"A {color_name} triangle on a white background"
            
        dataset.append({"image": img, "text": caption})
        
        # Print the first few samples to see what we are generating
        if i < 5:
            print(f"Sample {i+1}: {caption}")
            
    return dataset

# ==========================================
# 3. Preprocess Data (Extract Latents & Embeddings)
# ==========================================
def prepare_dataset_features(dataset, model_path):
    """
    Uses the VAE to encode images into latents.
    Uses CLIP Text Encoder to encode texts into hidden states.
    Doing this BEFORE the training loop saves massive amounts of GPU/TPU memory.
    """
    print("Loading VAE and Text Encoder for feature extraction...")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = FlaxCLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", dtype=jnp.float32)
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(model_path, subfolder="vae", dtype=jnp.float32)
    
    all_latents = []
    all_embeddings = []
    
    for item in dataset:
        # 3.1 Encode Text
        text_inputs = tokenizer(
            item["text"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="np"
        )
        # Get embeddings from the text encoder
        text_embeds = text_encoder(text_inputs.input_ids)[0]
        all_embeddings.append(text_embeds)
        
        # 3.2 Encode Image
        # Convert PIL Image to normalized numpy array [-1, 1]
        img_np = np.array(item["image"]).astype(np.float32) / 127.5 - 1.0
        # Transpose to channel-first (1, 3, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))[None, ...] 
        
        # Get latents from VAE
        vae_outputs = vae.apply({"params": vae_params}, img_np, method=vae.encode)
        # Sample from the distribution and scale it
        latents = vae_outputs.latent_dist.sample(jax.random.PRNGKey(0))
        
        # [CRITICAL FIX]: Transpose from (B, H, W, C) to (B, C, H, W) to match UNet's expected input format
        latents = jnp.transpose(latents, (0, 3, 1, 2))
        
        latents = latents * vae.config.scaling_factor
        all_latents.append(latents)
        
    # Concatenate all features into large JAX arrays
    latents_array = jnp.concatenate(all_latents, axis=0)
    embeddings_array = jnp.concatenate(all_embeddings, axis=0)
    
    print(f"Extracted Latents shape: {latents_array.shape}")
    print(f"Extracted Embeddings shape: {embeddings_array.shape}")
    
    return latents_array, embeddings_array

# ==========================================
# 4. Initialize UNet and Scheduler
# ==========================================
print("Initializing Scheduler and UNet...")
noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(
    config.pretrained_model_name_or_path, subfolder="scheduler"
)

# [MEMORY OOM FIX]: Load UNet explicitly in bfloat16. 
# This cuts memory consumption by 50% making it fit perfectly in 16GB TPU HBM!
unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    config.pretrained_model_name_or_path, subfolder="unet", dtype=jnp.bfloat16
)

# ==========================================
# 5. Set up Optimizer and Train State
# ==========================================
tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=1e-2)

# TrainState encapsulates model parameters and optimizer state
state = train_state.TrainState.create(
    apply_fn=unet.apply,
    params=unet_params,
    tx=tx,
)

# REPLICATE STATE FOR TPU: Copy the model state to all available TPU cores (8 on Kaggle TPU v5e-8)
num_devices = jax.device_count()
print(f"Number of available devices (TPU cores): {num_devices}")
state = jax_utils.replicate(state)

# ==========================================
# 6. Define PMAP-compiled Train Step (TPU Parallelism)
# ==========================================
@functools.partial(jax.pmap, axis_name='batch')
def train_step(state, batch_latents, batch_embeddings, train_rng):
    """
    Single training step.
    Compiled with @jax.pmap to run in parallel across all 8 TPU cores!
    """
    # Split RNG for noise generation and timestep sampling
    sample_rng, timestep_rng = jax.random.split(train_rng, 2)

    def compute_loss(params):
        # 1. Generate random Gaussian noise to add to the latents (Keep noise in float32 initially)
        noise = jax.random.normal(sample_rng, batch_latents.shape)
        
        # 2. Sample a random timestep for each image in the batch
        bsz = batch_latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            (bsz,),
            0,
            noise_scheduler.config.num_train_timesteps,
        )

        # 3. Forward diffusion: add noise to latents based on sampled timesteps
        noisy_latents = noise_scheduler.add_noise(
            noise_scheduler_state, batch_latents, noise, timesteps
        )

        # [CRITICAL MEMORY FIX]: Cast inputs to bfloat16 before feeding to the UNet
        noisy_latents_bf16 = noisy_latents.astype(jnp.bfloat16)
        batch_embeddings_bf16 = batch_embeddings.astype(jnp.bfloat16)

        # 4. Predict the noise residual using the UNet (Computation happens in bf16)
        model_pred = unet.apply(
            {"params": params},
            noisy_latents_bf16,
            timesteps,
            batch_embeddings_bf16,
        ).sample

        # 5. Calculate MSE loss between predicted noise and actual noise
        # [NOTE]: Cast prediction back to float32 to calculate Loss. This prevents gradient underflow/overflow.
        loss = jnp.mean((model_pred.astype(jnp.float32) - noise) ** 2)
        return loss

    # Compute loss and gradients simultaneously
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    
    # Cross-device gradient averaging (Crucial for multi-core TPU training)
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    # Update model parameters using the optimizer
    state = state.apply_gradients(grads=grads)
    
    return state, loss

# ==========================================
# 7. Main Training Loop
# ==========================================
if __name__ == "__main__":
    # Generate and process data
    raw_dataset = generate_synthetic_dataset(config.num_synthetic_samples)
    train_latents, train_embeddings = prepare_dataset_features(raw_dataset, config.pretrained_model_name_or_path)
    
    print(f"Starting training loop on {num_devices} TPU cores...")
    rng = jax.random.PRNGKey(config.seed)
    
    # Simple data generator for batching
    num_batches = len(train_latents) // config.batch_size
    batch_size_per_device = config.batch_size // num_devices
    
    for step in range(config.num_train_steps):
        # Split RNG keys: one for generating step RNGs, and one for the loop
        rng, step_rng = jax.random.split(rng, 2)
        # Create a separate RNG key for each TPU core
        step_rngs = jax.random.split(step_rng, num_devices)
        
        # Get current batch (cycling through the dataset)
        batch_idx = step % num_batches
        start_idx = batch_idx * config.batch_size
        end_idx = start_idx + config.batch_size
        
        b_latents = train_latents[start_idx:end_idx]
        b_embeddings = train_embeddings[start_idx:end_idx]
        
        # RESHAPE FOR TPU: (batch_size, ...) -> (num_devices, batch_size_per_device, ...)
        b_latents = b_latents.reshape((num_devices, batch_size_per_device) + b_latents.shape[1:])
        b_embeddings = b_embeddings.reshape((num_devices, batch_size_per_device) + b_embeddings.shape[1:])
        
        # Execute one training step across all TPU cores
        state, loss = train_step(state, b_latents, b_embeddings, step_rngs)
        
        if step % 10 == 0 or step == config.num_train_steps - 1:
            # Loss is replicated across devices, just take the 0th item
            # use jax.device_get to pull the scalar back to CPU for printing
            print(f"Step {step:04d} | Loss: {jax.device_get(loss[0]):.4f}")

    print("Training finished successfully!")
    
    # Optional: Save weights (unreplicate the state first!)
    # state_to_save = jax_utils.unreplicate(state)
    # from flax.training import checkpoints
    # checkpoints.save_checkpoint(ckpt_dir="kaggle/working/output_model", target=state_to_save, step=config.num_train_steps, keep=1)
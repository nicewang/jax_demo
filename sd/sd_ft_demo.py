import os
# CRITICAL: Prevent JAX from pre-allocating all TPU memory!
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
from datasets import load_dataset
from huggingface_hub import login
import argparse

def get_hf_token():
    """Attempts to read the Hugging Face token from Kaggle Secrets, then local properties."""
    # 1. First, try to get the token from Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("HF_TOKEN")
        if token:
            print("Successfully loaded HF_TOKEN from Kaggle Secrets.")
            return token
    except Exception as e:
        print("Kaggle Secrets not available or HF_TOKEN not found. Trying settings.properties...")

    # 2. Fallback to settings.properties if Kaggle Secrets fails
    filepath = "settings.properties"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                # Ignore empty lines and comments
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip().upper() == "HF_TOKEN":
                            print("Successfully loaded HF_TOKEN from settings.properties.")
                            return value.strip()
    return ""

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
class TrainConfig:
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    dataset_name = "lambdalabs/pokemon-blip-captions"
    
    # [CRITICAL FIX]: Read your Hugging Face token automatically
    hf_token = get_hf_token() 
    
    # On Kaggle TPU v5e-8, there are 8 cores. Batch size must be divisible by device count.
    batch_size = 8 
    learning_rate = 1e-4
    num_train_steps = 50 # 50 steps is enough to see the loss drop for a sanity check
    seed = 42
    
    # Using 8 samples instead of 5 because we have 8 TPU cores. 
    # Data size must be divisible by the global batch size (8).
    num_samples_to_test = 8 

config = TrainConfig()

# Automatically login to Hugging Face if a token is provided
if config.hf_token:
    print("Logging into Hugging Face Hub...")
    login(token=config.hf_token)
else:
    print("WARNING: No HF token provided. Dataset download might fail if it's gated.")

# ==========================================
# 2. Load Real Dataset (Pokemon)
# ==========================================
def load_real_dataset(dataset_name, num_samples):
    """
    Downloads the Pokemon dataset from Hugging Face and extracts a small subset for testing.
    """
    print(f"Downloading and loading '{dataset_name}' from Hugging Face...")
    # Load the training split
    dataset = load_dataset(dataset_name, split="train")
    
    # Select only the first 'num_samples' for our quick test
    dataset = dataset.select(range(num_samples))
    
    processed_dataset = []
    print(f"Processing {num_samples} images (Resizing to 512x512 and converting to RGB)...")
    
    for i, item in enumerate(dataset):
        # The dataset provides a PIL Image in the "image" column and text in "text"
        img = item["image"]
        caption = item["text"]
        
        # Ensure image is RGB (removes alpha channel if PNG) and resize to 512x512
        img = img.convert("RGB").resize((512, 512))
        
        processed_dataset.append({"image": img, "text": caption})
        print(f"Sample {i+1} caption: '{caption}'")
        
    return processed_dataset

# ==========================================
# 3. Preprocess Data (Extract Latents & Embeddings)
# ==========================================
def prepare_dataset_features(dataset, model_path):
    print("Loading VAE and Text Encoder on CPU to prevent TPU OOM...")
    
    # [ULTIMATE MEMORY FIX 1]: Load VAE and Text Encoder entirely on the CPU!
    cpu_device = jax.devices("cpu")[0]
    
    with jax.default_device(cpu_device):
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = FlaxCLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", dtype=jnp.float32, from_pt=True)
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(model_path, subfolder="vae", dtype=jnp.float32, from_pt=True)
        
        all_latents = []
        all_embeddings = []
        
        for item in dataset:
            # Encode Text
            text_inputs = tokenizer(
                item["text"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="np"
            )
            text_embeds = text_encoder(text_inputs.input_ids)[0]
            all_embeddings.append(np.array(text_embeds))
            
            # Encode Image
            img_np = np.array(item["image"]).astype(np.float32) / 127.5 - 1.0
            img_np = np.transpose(img_np, (2, 0, 1))[None, ...] 
            
            vae_outputs = vae.apply({"params": vae_params}, img_np, method=vae.encode)
            latents = vae_outputs.latent_dist.sample(jax.random.PRNGKey(0))
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * vae.config.scaling_factor
            
            all_latents.append(np.array(latents))
            
        latents_array = np.concatenate(all_latents, axis=0)
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        
    print(f"Extracted Latents shape: {latents_array.shape}")
    print(f"Extracted Embeddings shape: {embeddings_array.shape}")
    
    # Free up CPU memory
    import gc
    del vae, vae_params, text_encoder, tokenizer, all_latents, all_embeddings
    gc.collect()
    
    return latents_array, embeddings_array

# ==========================================
# 4. Initialize UNet and Scheduler
# ==========================================
print("Initializing Scheduler...")
noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(
    config.pretrained_model_name_or_path, subfolder="scheduler"
)

print("Initializing UNet and Optimizer on CPU first to prevent TPU OOM...")
cpu_device = jax.devices("cpu")[0]

# [ULTIMATE MEMORY FIX 2]: Load UNet on CPU first! 
with jax.default_device(cpu_device):
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", dtype=jnp.bfloat16, from_pt=True
    )

    # Adafactor optimizer to save huge TPU memory
    tx = optax.adafactor(
        learning_rate=config.learning_rate,
        multiply_by_parameter_scale=False
    )

    state = train_state.TrainState.create(
        apply_fn=unet.apply,
        params=unet_params,
        tx=tx,
    )

# ==========================================
# 5. Replicate State for TPU Parallelism
# ==========================================
num_devices = jax.device_count()
print(f"Replicating clean model parameters to {num_devices} TPU cores...")
state = jax_utils.replicate(state)

# ==========================================
# 6. Define PMAP-compiled Train Step (TPU Parallelism)
# ==========================================
@functools.partial(jax.pmap, axis_name='batch')
def train_step(state, batch_latents, batch_embeddings, train_rng):
    sample_rng, timestep_rng = jax.random.split(train_rng, 2)

    def compute_loss(params):
        noise = jax.random.normal(sample_rng, batch_latents.shape)
        bsz = batch_latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng, (bsz,), 0, noise_scheduler.config.num_train_timesteps,
        )

        noisy_latents = noise_scheduler.add_noise(
            noise_scheduler_state, batch_latents, noise, timesteps
        )

        noisy_latents_bf16 = noisy_latents.astype(jnp.bfloat16)
        batch_embeddings_bf16 = batch_embeddings.astype(jnp.bfloat16)

        model_pred = unet.apply(
            {"params": params},
            noisy_latents_bf16,
            timesteps,
            batch_embeddings_bf16,
        ).sample

        loss = jnp.mean((model_pred.astype(jnp.float32) - noise) ** 2)
        return loss

    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    state = state.apply_gradients(grads=grads)
    return state, loss

# ==========================================
# 7. Main Training Loop
# ==========================================
if __name__ == "__main__":
    # Load Real Dataset instead of synthetic
    raw_dataset = load_real_dataset(config.dataset_name, config.num_samples_to_test)
    train_latents, train_embeddings = prepare_dataset_features(raw_dataset, config.pretrained_model_name_or_path)
    
    print(f"Starting training loop on {num_devices} TPU cores...")
    rng = jax.random.PRNGKey(config.seed)
    
    num_batches = len(train_latents) // config.batch_size
    batch_size_per_device = config.batch_size // num_devices
    
    for step in range(config.num_train_steps):
        rng, step_rng = jax.random.split(rng, 2)
        step_rngs = jax.random.split(step_rng, num_devices)
        
        batch_idx = step % num_batches
        start_idx = batch_idx * config.batch_size
        end_idx = start_idx + config.batch_size
        
        b_latents = train_latents[start_idx:end_idx]
        b_embeddings = train_embeddings[start_idx:end_idx]
        
        b_latents = b_latents.reshape((num_devices, batch_size_per_device) + b_latents.shape[1:])
        b_embeddings = b_embeddings.reshape((num_devices, batch_size_per_device) + b_embeddings.shape[1:])
        
        state, loss = train_step(state, b_latents, b_embeddings, step_rngs)
        
        if step % 10 == 0 or step == config.num_train_steps - 1:
            print(f"Step {step:04d} | Loss: {jax.device_get(loss[0]):.4f}")

    print("Training finished successfully!")
    
    state_to_save = jax_utils.unreplicate(state)
    from flax.training import checkpoints
    checkpoints.save_checkpoint(ckpt_dir="/kaggle/working/model_pokemon", target=state_to_save, step=config.num_train_steps, keep=1)
    print("Model weights successfully saved to /kaggle/working/model_pokemon !")
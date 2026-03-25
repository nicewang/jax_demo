import os

# ==============================================================================
# CRITICAL C++ FIXES FOR KAGGLE TPU KERNEL RESTARTS
# ==============================================================================
# 1. Prevent JAX from pre-allocating all TPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 2. Prevent "Thread handle creation failed" (EAGAIN/11) C++ crashes.
# Kaggle containers have a strict thread limit. XLA/TSL tries to aggressively 
# allocate huge thread pools for operations. We MUST limit them here.
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

import gc
import json
import time
import psutil
import argparse
import functools
import numpy as np

# 3. Prevent matplotlib from silently starting background GUI threads
# (Moved to standalone plotting function to avoid interfering with JAX)

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from flax import jax_utils

from diffusers import FlaxUNet2DConditionModel, FlaxDDPMScheduler, FlaxAutoencoderKL
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from datasets import load_dataset
from huggingface_hub import login


def get_hf_token():
    """Attempts to read the Hugging Face token from Kaggle Secrets, then local properties."""
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("HF_TOKEN")
        if token:
            print("Successfully loaded HF_TOKEN from Kaggle Secrets.")
            return token
    except Exception as e:
        print("Kaggle Secrets not available or HF_TOKEN not found. Trying settings.properties...")

    filepath = "settings.properties"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip().upper() == "HF_TOKEN":
                        print("Successfully loaded HF_TOKEN from settings.properties.")
                        return value.strip()
    return ""


class TrainConfig:
    """Configuration class for hyperparameters and paths."""
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    dataset_name = "lambdalabs/naruto-blip-captions"
    hf_token = get_hf_token() 
    
    # On Kaggle TPU v5e-8, there are 8 cores. Batch size must be divisible by device count.
    batch_size = 8 
    learning_rate = 1e-4
    num_train_steps = 50 
    seed = 42
    num_samples_to_test = 8 
    
    output_dir = "/kaggle/working/model_naruto"
    metrics_dir = "/kaggle/working"


def create_train_step(unet, noise_scheduler, noise_scheduler_state):
    """
    Factory function to isolate JAX pmap from class scope.
    This prevents memory leaks during compilation.
    """
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

    return train_step


class SDFineTuner:
    """
    Encapsulates the Stable Diffusion fine-tuning process on TPUs using JAX/Flax.
    Tracks performance metrics (Algorithm, CPU, RAM, TPU Mem) and visualizes them.
    """
    def __init__(self, config=None):
        self.config = config or TrainConfig()
        
        if self.config.hf_token:
            print("Logging into Hugging Face Hub...")
            login(token=self.config.hf_token)
        else:
            print("WARNING: No HF token provided. Dataset download might fail if it's gated.")

        # Ensure output directories exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.metrics_dir, exist_ok=True)

    def load_real_dataset(self):
        """Downloads and processes the image dataset."""
        print(f"Downloading and loading '{self.config.dataset_name}' from Hugging Face...")
        dataset = load_dataset(self.config.dataset_name, split="train", token=self.config.hf_token)
        dataset = dataset.select(range(self.config.num_samples_to_test))
        
        processed_dataset = []
        print(f"Processing {self.config.num_samples_to_test} images (Resizing to 512x512 and converting to RGB)...")
        
        for i, item in enumerate(dataset):
            img = item["image"].convert("RGB").resize((512, 512))
            caption = item["text"]
            processed_dataset.append({"image": img, "text": caption})
        
        return processed_dataset

    def prepare_dataset_features(self, dataset):
        """Extracts latents and text embeddings on CPU to prevent TPU OOM."""
        print("Loading VAE and Text Encoder on CPU to prevent TPU OOM...")
        cpu_device = jax.devices("cpu")[0]
        
        with jax.default_device(cpu_device):
            tokenizer = CLIPTokenizer.from_pretrained(self.config.pretrained_model_name_or_path, subfolder="tokenizer")
            text_encoder = FlaxCLIPTextModel.from_pretrained(self.config.pretrained_model_name_or_path, subfolder="text_encoder", dtype=jnp.float32, from_pt=True)
            vae, vae_params = FlaxAutoencoderKL.from_pretrained(self.config.pretrained_model_name_or_path, subfolder="vae", dtype=jnp.float32, from_pt=True)
            
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
        del vae, vae_params, text_encoder, tokenizer, all_latents, all_embeddings
        gc.collect()
        
        return latents_array, embeddings_array

    def fine_tune(self):
        """Main training workflow."""
        # 1. Data Preparation
        raw_dataset = self.load_real_dataset()
        train_latents, train_embeddings = self.prepare_dataset_features(raw_dataset)
        
        # Free up huge amounts of System RAM before UNet initializes.
        del raw_dataset
        gc.collect()
        
        # 2. Model Initialization
        print("Initializing Scheduler...")
        noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )

        print("Initializing UNet and Optimizer on CPU first to prevent TPU OOM...")
        cpu_device = jax.devices("cpu")[0]
        
        with jax.default_device(cpu_device):
            unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
                self.config.pretrained_model_name_or_path, subfolder="unet", dtype=jnp.bfloat16, from_pt=True
            )
            # Adafactor optimizer
            tx = optax.adafactor(
                learning_rate=self.config.learning_rate,
                multiply_by_parameter_scale=False
            )
            state = train_state.TrainState.create(
                apply_fn=unet.apply,
                params=unet_params,
                tx=tx,
            )

        # 3. Replicate State for TPU Parallelism
        num_devices = jax.device_count()
        print(f"Replicating clean model parameters to {num_devices} TPU cores...")
        state = jax_utils.replicate(state)

        # 4. Bind the factory function strictly without memory leaks
        train_step_fn = create_train_step(unet, noise_scheduler, noise_scheduler_state)

        # 5. Training Loop
        print(f"Starting training loop on {num_devices} TPU cores...")
        rng = jax.random.PRNGKey(self.config.seed)
        
        num_batches = len(train_latents) // self.config.batch_size
        batch_size_per_device = self.config.batch_size // num_devices
        
        # Track metrics including TPU
        history = {'step': [], 'loss': [], 'cpu_percent': [], 'ram_gb': [], 'tpu_mem_gb': []}
        psutil.cpu_percent(interval=None) # Initialize baseline

        for step in range(self.config.num_train_steps):
            rng, step_rng = jax.random.split(rng, 2)
            step_rngs = jax.random.split(step_rng, num_devices)
            
            batch_idx = step % num_batches
            start_idx = batch_idx * self.config.batch_size
            end_idx = start_idx + self.config.batch_size
            
            b_latents = train_latents[start_idx:end_idx]
            b_embeddings = train_embeddings[start_idx:end_idx]
            
            b_latents = b_latents.reshape((num_devices, batch_size_per_device) + b_latents.shape[1:])
            b_embeddings = b_embeddings.reshape((num_devices, batch_size_per_device) + b_embeddings.shape[1:])
            
            state, loss = train_step_fn(state, b_latents, b_embeddings, step_rngs)
            
            # --- Metrics Collection ---
            loss_val = float(jax.device_get(loss[0]))
            cpu_util = psutil.cpu_percent()
            ram_used = psutil.virtual_memory().used / (1024**3)
            
            # Safely get TPU memory now that thread exhaustion is fixed
            try:
                tpu_stats = jax.local_devices()[0].memory_stats()
                tpu_mem = tpu_stats.get('bytes_in_use', 0) / (1024**3)
            except Exception:
                tpu_mem = 0.0

            history['step'].append(step)
            history['loss'].append(loss_val)
            history['cpu_percent'].append(cpu_util)
            history['ram_gb'].append(ram_used)
            history['tpu_mem_gb'].append(tpu_mem)
            
            if step % 10 == 0 or step == self.config.num_train_steps - 1:
                print(f"Step {step:04d} | Loss: {loss_val:.4f} | CPU: {cpu_util}% | RAM: {ram_used:.2f}GB | TPU: {tpu_mem:.2f}GB")

        print("Training finished successfully!")
        
        # 6. Save Model
        unreplicated = jax_utils.unreplicate(state)
        unet_params_to_save = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16),
            unreplicated.params
        )

        checkpoints.save_checkpoint(
            ckpt_dir=self.config.output_dir,
            target={"params": unet_params_to_save},
            step=self.config.num_train_steps,
            keep=1,
            overwrite=True,
        )
        print(f"Model weights successfully saved to {self.config.output_dir} !")

        # Save raw JSON data for future analysis
        json_path = os.path.join(self.config.metrics_dir, "training_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Metrics raw data saved to: {json_path}")
        
        return history

# ==========================================
# Standalone Plotting Function
# ==========================================
def plot_training_metrics(history, metrics_dir):
    """Generates and saves a plot of the training metrics after fine-tuning completes."""
    print("Generating and saving training metrics plot...")
    
    # Import matplotlib ONLY here, after all JAX/XLA operations are fully complete
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    steps = history['step']
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stable Diffusion Fine-Tuning Performance Metrics', fontsize=16)

    # 1. Algorithm Metric: Loss
    axs[0, 0].plot(steps, history['loss'], color='tab:red', marker='o', markersize=3)
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Step')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)

    # 2. Performance Metric: CPU Usage
    axs[0, 1].plot(steps, history['cpu_percent'], color='tab:blue')
    axs[0, 1].set_title('CPU Usage (%)')
    axs[0, 1].set_xlabel('Step')
    axs[0, 1].set_ylabel('Percentage (%)')
    axs[0, 1].grid(True)
    axs[0, 1].set_ylim(0, 100)

    # 3. Performance Metric: System RAM
    axs[1, 0].plot(steps, history['ram_gb'], color='tab:green')
    axs[1, 0].set_title('System RAM Usage (GB)')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('GB')
    axs[1, 0].grid(True)

    # 4. Performance Metric: TPU Memory
    axs[1, 1].plot(steps, history['tpu_mem_gb'], color='tab:purple')
    axs[1, 1].set_title('TPU Memory Usage (GB) - Device 0')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('GB')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(metrics_dir, "training_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Metrics plot saved to: {plot_path}")

# ==========================================
# Example usage: 1-line invocation
# ==========================================
if __name__ == "__main__":
    config = TrainConfig()
    
    # 1. Run the fine-tuner (which tracks metrics and returns the dictionary)
    metrics_history = SDFineTuner(config).fine_tune()
    
    # 2. Generate the plot safely after training is entirely complete
    plot_training_metrics(metrics_history, config.metrics_dir)
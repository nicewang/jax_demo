import os
# Prevent JAX from pre-allocating all TPU memory, just like in training!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from diffusers import FlaxStableDiffusionPipeline
from flax.training import checkpoints
from flax import jax_utils
from IPython.display import display

# 1. Use the real SD 1.5 model, and add from_pt=True to automatically convert weights
model_id = "runwayml/stable-diffusion-v1-5"
print("Loading the base pipeline on CPU to prevent TPU OOM...")

# [ULTIMATE MEMORY FIX]: Force JAX to load and convert the entire pipeline on the CPU!
cpu_device = jax.devices("cpu")[0]

with jax.default_device(cpu_device):
    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(model_id, dtype=jnp.bfloat16, from_pt=True)

    print("Replacing with our fine-tuned UNet weights...")
    raw_checkpoint = checkpoints.restore_checkpoint(ckpt_dir="/kaggle/working/output_model", target=None)
    
    # Force the restored checkpoint into bfloat16 to ensure it doesn't inflate memory
    unet_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), raw_checkpoint['params'])
    params["unet"] = unet_params

print("Replicating clean parameters to all 8 TPU cores...")
p_params = jax_utils.replicate(params)

# 2. Define the prompt you want to test (can be changed to words from your fine-tuning data)
prompt = "A red circle on a white background"
print(f"Testing prompt: '{prompt}'")

prompts = [prompt] * jax.device_count()
prompt_ids = pipe.prepare_inputs(prompts)
p_prompt_ids = jax_utils.replicate(prompt_ids)

prng_seed = jax.random.split(jax.random.PRNGKey(42), jax.device_count())

print("Generating image via TPU, this might take a moment...")
output = pipe(
    prompt_ids=p_prompt_ids,
    params=p_params,
    prng_seed=prng_seed,
    num_inference_steps=50, # 50 steps are recommended for the real model
    jit=True
)

print("Generation complete!")
images = pipe.numpy_to_pil(np.asarray(output.images[0]))
display(images[0])
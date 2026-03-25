import os
# Prevent JAX from pre-allocating all TPU memory!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np
import gc # Import garbage collection
from diffusers import FlaxStableDiffusionPipeline
from flax.training import checkpoints
from flax import jax_utils
from flax.core import unfreeze, freeze
from IPython.display import display

model_id = "runwayml/stable-diffusion-v1-5"
print("Loading the base pipeline on CPU to prevent TPU OOM...")

cpu_device = jax.devices("cpu")[0]

with jax.default_device(cpu_device):
    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(model_id, dtype=jnp.bfloat16, from_pt=True)
    
    # Unfreeze the dictionary so we can modify it safely
    params = unfreeze(params)

    print("Freeing original UNet from CPU RAM to prevent kernel crash...")
    del params["unet"]
    gc.collect()

    print("Replacing with our fine-tuned UNet weights...")
    raw_checkpoint = checkpoints.restore_checkpoint(ckpt_dir="/kaggle/working/model_naruto", target=None)
    
    print("Casting UNet to bfloat16...")
    # Directly assign and cast to bfloat16
    params["unet"] = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), raw_checkpoint['params'])
    
    print("Freeing temporary checkpoint variables...")
    del raw_checkpoint
    gc.collect()

    print("Casting remaining components to bfloat16...")
    for key in params.keys():
        if key != "unet":
            params[key] = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.bfloat16), params[key])
    gc.collect()

print("Replicating parameters to TPUs component by component to save RAM...")
p_params = {}

# [ULTIMATE MEMORY FIX]: Replicate one component at a time, and delete the CPU copy immediately!
for key in list(params.keys()):
    print(f" -> Replicating {key} to 8 TPUs...")
    p_params[key] = jax_utils.replicate(params[key])
    
    # Delete the CPU version immediately after it is sent to TPUs to prevent RAM spikes
    del params[key] 
    gc.collect()

# Freeze the dictionary back to its required Flax format
p_params = freeze(p_params)

prompt = "A drawing of Naruto Uzumaki"
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
    num_inference_steps=50,
    jit=True
)

print("Generation complete!")
images = pipe.numpy_to_pil(np.asarray(output.images[0]))
display(images[0])
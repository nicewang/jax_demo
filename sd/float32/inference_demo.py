import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]   = "platform"

import gc
import jax
import jax.numpy as jnp
import numpy as np
from diffusers import FlaxStableDiffusionPipeline
from flax import jax_utils
from flax.core import freeze, unfreeze
from flax.training import checkpoints
from IPython.display import display

# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — Nuke-level cleanup: Forcefully release residual JAX memory (CRITICAL!)
# ──────────────────────────────────────────────────────────────────────────────
print("Clearing JAX compiler caches from training phase...")
# This forces JAX to clear all cached compiled graphs from the fine-tuning 
# phase, freeing up precious TPU HBM for the inference compilation.
jax.clear_caches()
gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# Device setup — use only 1 TPU core for single-image inference.
# ──────────────────────────────────────────────────────────────────────────────
cpu_device = jax.devices("cpu")[0]
single_tpu = jax.devices()[:1]

print(f"CPU: {cpu_device}")
print(f"TPU: {single_tpu}")

MODEL_ID       = "runwayml/stable-diffusion-v1-5"
CHECKPOINT_DIR = "/kaggle/working/model_naruto_float32"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load the base pipeline on CPU in float32.
# ──────────────────────────────────────────────────────────────────────────────
print("Loading base pipeline on CPU in float32…")
with jax.default_device(cpu_device):
    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=jnp.float32,  # Strictly keep float32 precision
        from_pt=True,
        safety_checker=None,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Swap in the fine-tuned UNet weights.
# ──────────────────────────────────────────────────────────────────────────────
print("Replacing base UNet with fine-tuned Naruto weights…")
with jax.default_device(cpu_device):
    params = unfreeze(params)

    # Free the base UNet first so we never hold two UNets at once.
    del params["unet"]
    gc.collect()

    # Load the checkpoint. target=None → raw dict
    raw_ckpt = checkpoints.restore_checkpoint(ckpt_dir=CHECKPOINT_DIR, target=None)

    # Assign directly to perfectly preserve your float32 precision.
    params["unet"] = raw_ckpt["params"]

    del raw_ckpt
    gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Move parameters to the single TPU core.
# ──────────────────────────────────────────────────────────────────────────────
print("Replicating parameters to TPU core 0…")
p_params = {}
for key in list(params.keys()):
    print(f"  → {key}")
    p_params[key] = jax_utils.replicate(params[key], devices=single_tpu)
    jax.block_until_ready(p_params[key])   # wait for DMA transfer to finish
    del params[key]
    gc.collect()

p_params = freeze(p_params)
print("All parameters on TPU.")

# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Run inference.
# ──────────────────────────────────────────────────────────────────────────────
PROMPT = "A drawing of Naruto Uzumaki"
print(f"\nPrompt: '{PROMPT}'")

prompts      = [PROMPT] * len(single_tpu)
prompt_ids   = pipe.prepare_inputs(prompts)
p_prompt_ids = jax_utils.replicate(prompt_ids, devices=single_tpu)
prng_seed    = jax.random.split(jax.random.PRNGKey(42), len(single_tpu))

print("Compiling and generating image (this might take a moment)...")
output = pipe(
    prompt_ids=p_prompt_ids,
    params=p_params,
    prng_seed=prng_seed,
    num_inference_steps=50,
    jit=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Convert to PIL and display.
# ──────────────────────────────────────────────────────────────────────────────
print("Transferring result to host…")
jax.block_until_ready(output.images)

H, W, C    = output.images.shape[-3:]
images_np  = np.asarray(output.images.reshape(-1, H, W, C))
images_pil = pipe.numpy_to_pil(images_np)

print("Done!")
display(images_pil[0])
import os

import gc

import jax
import jax.numpy as jnp
import numpy as np
from diffusers import FlaxStableDiffusionPipeline
from flax import jax_utils
from flax.core import freeze, unfreeze
from flax.training import checkpoints
from IPython.display import display

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]   = "platform"

# ──────────────────────────────────────────────────────────────────────────────
# Device setup — use only 1 TPU core for single-image inference.
# Using all 8 cores would force jax_utils.replicate() to create 8 copies of
# every parameter tensor in CPU RAM before the first tensor reaches the TPU.
# ──────────────────────────────────────────────────────────────────────────────
cpu_device = jax.devices("cpu")[0]
single_tpu = jax.devices()[:1]

print(f"CPU: {cpu_device}")
print(f"TPU: {single_tpu}")

MODEL_ID       = "runwayml/stable-diffusion-v1-5"
CHECKPOINT_DIR = "/kaggle/working/model_naruto_float32"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load the base pipeline on CPU.
#
# dtype = float32.
# Loading on CPU keeps everything out of TPU HBM until we explicitly move it.
# ──────────────────────────────────────────────────────────────────────────────
print("Loading base pipeline on CPU…")
with jax.default_device(cpu_device):
    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=jnp.float32,
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

    # Load the checkpoint.  target=None → raw dict, no pytree shape hint needed.
    raw_ckpt = checkpoints.restore_checkpoint(ckpt_dir=CHECKPOINT_DIR, target=None)

    # No cast required — assign directly.
    params["unet"] = raw_ckpt["params"]

    del raw_ckpt
    gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Verify dtypes for all components.
# ──────────────────────────────────────────────────────────────────────────────
print("Verifying component dtypes…")
with jax.default_device(cpu_device):
    for key in list(params.keys()):
        leaves = jax.tree_util.tree_leaves(params[key])
        if not leaves:
            continue
    gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Move parameters to the single TPU core.
#
# We process one component at a time.  After each replicate() call we call
# jax.block_until_ready() to force XLA to complete the DMA transfer before
# we delete the CPU-side source.  Without this, the async XLA runtime may hold
# a hidden reference and the CPU buffer will not be freed until much later.
# ──────────────────────────────────────────────────────────────────────────────
print("Replicating parameters to TPU core 0…")
p_params = {}
for key in list(params.keys()):
    print(f"  → {key}")
    p_params[key] = jax_utils.replicate(params[key], devices=single_tpu)
    jax.block_until_ready(p_params[key])   # wait for transfer to finish
    del params[key]
    gc.collect()

p_params = freeze(p_params)
print("All parameters on TPU.")

# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Run inference.
# ──────────────────────────────────────────────────────────────────────────────
PROMPT = "A drawing of Naruto Uzumaki"
print(f"\nPrompt: '{PROMPT}'")

prompts      = [PROMPT] * len(single_tpu)
prompt_ids   = pipe.prepare_inputs(prompts)
p_prompt_ids = jax_utils.replicate(prompt_ids, devices=single_tpu)
prng_seed    = jax.random.split(jax.random.PRNGKey(42), len(single_tpu))

output = pipe(
    prompt_ids=p_prompt_ids,
    params=p_params,
    prng_seed=prng_seed,
    num_inference_steps=50,
    jit=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Convert to PIL and display.
#
# output.images shape: (num_devices, batch_per_device, H, W, C)
#                    = (1, 1, 512, 512, 3) with our single-device setup.
#
# block_until_ready() ensures the TPU computation is complete before we
# initiate the device→host copy.  reshape(-1, H, W, C) collapses the
# leading (num_devices, batch) dims so numpy_to_pil gets the expected layout.
# ──────────────────────────────────────────────────────────────────────────────
print("Transferring result to host…")
jax.block_until_ready(output.images)

H, W, C    = output.images.shape[-3:]
images_np  = np.asarray(output.images.reshape(-1, H, W, C))
images_pil = pipe.numpy_to_pil(images_np)

print("Done!")
display(images_pil[0])
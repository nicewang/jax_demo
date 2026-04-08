"""
UAV Sim2Real Training Demo with MuJoCo XLA (MJX), JAX, Brax using OpenSource Dataset from Hugging Face.
Optimized drone reinforcement learning training script for Kaggle TPU v5e-8.
"""

import os
import warnings

# Suppress Brax deprecation warnings (since we are correctly using MJX backend)
warnings.filterwarnings("ignore", category=UserWarning)

# --- SAFE MEMORY ALLOCATION ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import mujoco
from mujoco import mjx

# Use PipelineEnv (required by Brax 0.14.x)
from brax.envs.base import State, PipelineEnv
from brax.io import mjcf as brax_mjcf
from brax.training.agents.ppo import train as ppo
from brax.io import model as brax_model
from brax.io import html

import pandas as pd
import numpy as np
import ast
import time
import glob
from huggingface_hub import HfApi, snapshot_download, login

# ==========================================
# 1. Data Processing and Batch Loading Logic
# ==========================================
def parse_dataframe(df, num_points):
    """Helper function: Parse coordinate points from DataFrames with different structures"""
    df = df.head(num_points)
    cols = df.columns.tolist()

    if 'x' in cols and 'y' in cols and 'z' in cols:
        return np.column_stack((df['x'], df['y'], df['z']))
    elif 'tx' in cols and 'ty' in cols and 'tz' in cols:
        return np.column_stack((df['tx'], df['ty'], df['tz']))
    elif 'position' in cols:
        positions = df['position'].tolist()
        if isinstance(positions[0], str):
            positions = [ast.literal_eval(p) for p in positions]
        return np.array(positions)
    else:
        x_col = next((c for c in cols if c.lower() in ['x', 'tx'] or c.lower().endswith('.x') or c.lower().endswith('_x')), None)
        y_col = next((c for c in cols if c.lower() in ['y', 'ty'] or c.lower().endswith('.y') or c.lower().endswith('_y')), None)
        z_col = next((c for c in cols if c.lower() in ['z', 'tz'] or c.lower().endswith('.z') or c.lower().endswith('_z')), None)
        if x_col and y_col and z_col:
            return np.column_stack((df[x_col], df[y_col], df[z_col]))
        else:
            raise ValueError(f"Unrecognized column names. Available columns are: {cols}")

def get_csv_file_lists(repo_id, hf_token=None):
    """Get all CSV filenames in the repo and split them equally into two batches"""
    print(f"Fetching full file list from {repo_id}...")
    api = HfApi(token=hf_token)
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    csv_files = [f for f in all_files if f.endswith('.csv')]
    mid_point = len(csv_files) // 2
    batch1 = csv_files[:mid_point]
    batch2 = csv_files[mid_point:]
    print(f"Discovered a total of {len(csv_files)} CSV files.")
    print(f"Split into two batches: Batch 1 ({len(batch1)} files), Batch 2 ({len(batch2)} files).")
    return batch1, batch2

def download_batch_and_extract_demo(repo_id, file_list, hf_token, num_demo_points=4):
    """
    Download specified batch of files (multi-threaded), and extract a certain number of waypoints for Demo training.
    """
    print(f"\nStarting batch download ({len(file_list)} files in total)...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=file_list,
        token=hf_token,
        max_workers=8
    )
    local_csvs = glob.glob(os.path.join(local_dir, "**/*.csv"), recursive=True)
    local_csvs.sort()
    if not local_csvs:
        raise FileNotFoundError("Could not find downloaded CSV files locally!")
    print(f"Batch download complete! {len(local_csvs)} related files exist locally.")
    print(f"Extracting {num_demo_points} trajectory points from the first file ({os.path.basename(local_csvs[0])})...")
    df = pd.read_csv(local_csvs[0])
    waypoints = parse_dataframe(df, num_points=num_demo_points)
    print(f"Successfully extracted {len(waypoints)} waypoints!")
    return jnp.array(waypoints, dtype=jnp.float32)

# ==========================================
# 2. MuJoCo UAV (Quadrotor) Model Definition (XML)
# ==========================================
# Integrator changed to "Euler" to prevent XLA compiler memory explosion (OOM) on TPU.
UAV_XML = """
<mujoco model="quadrotor">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option gravity="0 0 -9.81" timestep="0.01" integrator="Euler"/>

  <default>
    <geom friction="1 0.1 0.1" margin="0.001" rgba="0.8 0.6 0.4 1"/>
    <joint damping="0.01"/>
  </default>

  <worldbody>
    <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>
    <geom type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>

    <!-- UAV Body -->
    <body name="uav" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="core" type="box" size="0.1 0.1 0.05" mass="1.0" rgba="0.2 0.5 0.8 1"/>
      <site name="rotor1" pos="0.1 0.1 0" size="0.05" rgba="1 0 0 1"/>
      <site name="rotor2" pos="-0.1 0.1 0" size="0.05" rgba="1 0 0 1"/>
      <site name="rotor3" pos="-0.1 -0.1 0" size="0.05" rgba="0 1 0 1"/>
      <site name="rotor4" pos="0.1 -0.1 0" size="0.05" rgba="0 1 0 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="m1" site="rotor1" ctrlrange="0 5" ctrllimited="true"/>
    <motor name="m2" site="rotor2" ctrlrange="0 5" ctrllimited="true"/>
    <motor name="m3" site="rotor3" ctrlrange="0 5" ctrllimited="true"/>
    <motor name="m4" site="rotor4" ctrlrange="0 5" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

# ==========================================
# 3. Brax/MJX RL Environment
# ==========================================
class UAVTrackingEnv(PipelineEnv):
    def __init__(self, waypoints):
        sys = brax_mjcf.loads(UAV_XML)
        super().__init__(sys=sys, backend='mjx', n_frames=1)
        self.waypoints = waypoints
        self.num_waypoints = waypoints.shape[0]

    def reset(self, rng: jnp.ndarray) -> State:
        rng, rng_state = jax.random.split(rng)
        
        # Spawn UAV near the first waypoint
        target_pos = self.waypoints[0]
        # 1.0m to Z axis to ensure it spawns safely above ground
        init_pos = target_pos + jnp.array([0.0, 0.0, 1.0])
        init_q = self.sys.init_q.at[:3].set(init_pos)
        
        pipeline_state = self.pipeline_init(
            init_q + jax.random.uniform(rng_state, (self.sys.q_size(),), minval=-0.1, maxval=0.1),
            jnp.zeros(self.sys.qd_size())
        )
        obs = self._get_obs(pipeline_state, target_idx=jnp.array(0))
        uav_pos = pipeline_state.q[:3]
        target_pos = self.waypoints[0]
        initial_distance = jnp.linalg.norm(uav_pos - target_pos)
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            metrics={"target_idx": jnp.array(0.0), "distance_to_target": initial_distance}
        )

    def step(self, state: State, action: jnp.ndarray) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        target_idx = state.metrics["target_idx"].astype(jnp.int32)

        uav_pos = pipeline_state.q[:3]
        target_pos = self.waypoints[target_idx]
        distance = jnp.linalg.norm(uav_pos - target_pos)

        # Reward formulation: negative distance to encourage closing the gap
        reward = -distance

        reached = distance < 0.2
        next_idx = jnp.minimum(target_idx + 1, self.num_waypoints - 1)
        target_idx = jnp.where(reached, next_idx, target_idx).astype(jnp.float32)

        # Termination conditions
        done = jnp.where(uav_pos[2] < 0.1, 1.0, 0.0) # Crashed into ground
        done = jnp.where(distance > 20.0, 1.0, done) # Flew too far away

        obs = self._get_obs(pipeline_state, target_idx.astype(jnp.int32))
        
        metrics = state.metrics.copy()
        metrics["target_idx"] = target_idx
        metrics["distance_to_target"] = distance
        
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics)

    def _get_obs(self, pipeline_state, target_idx) -> jnp.ndarray:
        uav_pos  = pipeline_state.q[:3]
        uav_quat = pipeline_state.q[3:7]
        uav_vel  = pipeline_state.qd[:3]
        target_pos = self.waypoints[target_idx]
        return jnp.concatenate([uav_pos, uav_quat, uav_vel, target_pos])

    @property
    def action_size(self) -> int: return 4
    @property
    def observation_size(self) -> int: return 13

# ==========================================
# 4. Main Flow
# ==========================================
def main():
    # Setup JAX Mesh for proper SPMD parallelization on TPU
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=('batch',))
    print(f"JAX Hardware Devices: {jax.device_count()} TPU cores")
    print(f"Mesh created: {mesh}")

    # --- GET HUGGING FACE TOKEN ---
    hf_token = None
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN_WRITE")
        print("Successfully retrieved HF_TOKEN_WRITE from Kaggle Secrets!")
    except Exception:
        print("Kaggle Secrets not detected, attempting in anonymous mode.")

    REPO_ID = "riotu-lab/Synthetic-UAV-Flight-Trajectories"

    batch1_files, batch2_files = get_csv_file_lists(REPO_ID, hf_token)
    stage1_start_time = time.time()

    # --- STAGE 1 ---

    # Global timer to measure end-to-end execution time
    pipeline_start_time = time.time()

    print("\n[STAGE 1] ---------------------------------------------")
    waypoints_batch1 = download_batch_and_extract_demo(REPO_ID, batch1_files, hf_token, num_demo_points=4)

    print("\nStarting Stage 1 PPO training (Based on Batch 1 data)...")
    env_1 = UAVTrackingEnv(waypoints_batch1)

    with mesh:
        make_inference_fn_1, params_1, _ = ppo.train(
            environment=env_1,
            # INCREASED: Give it enough steps to learn how to fly
            num_timesteps=2_000_000, 
            num_evals=5,
            reward_scaling=1.0,
            # INCREASED: 500 steps = 5 seconds physics time (gives it time to fly)
            episode_length=500,      
            normalize_observations=False, 
            action_repeat=1,
            unroll_length=10,        
            num_minibatches=16,       
            num_updates_per_batch=4,
            discounting=0.99,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=128,            
            batch_size=80,           
            seed=42,
        )
    print("Stage 1 training completed successfully.")

    # --- Memory Cleanup between stages ---
    print("\n[Memory Cleanup] Clearing JAX compilation cache...")
    jax.clear_caches()
    del env_1
    import gc
    gc.collect()
    print("Cache cleared.")

    # --- Forced Cooldown ---
    print("\n[API Protection Mechanism] ------------------------------------------")
    elapsed_time = time.time() - stage1_start_time
    wait_target = 310
    if elapsed_time < wait_target:
        sleep_duration = wait_target - elapsed_time
        print(f"Only {elapsed_time:.1f}s elapsed. Sleeping {sleep_duration:.1f}s to reset HF API quota...")
        time.sleep(sleep_duration)
        print("Wait over! API quota has been reset.")
    else:
        print(f"Stage 1 took {elapsed_time:.1f}s, safely exceeding the 5-minute limit. Continuing directly!")

    # --- STAGE 2 ---
    print("\n[STAGE 2] ---------------------------------------------")
    waypoints_batch2 = download_batch_and_extract_demo(REPO_ID, batch2_files, hf_token, num_demo_points=4)

    print("\nStarting Stage 2 Continual Learning...")
    env_2 = UAVTrackingEnv(waypoints_batch2)

    with mesh:
        make_inference_fn_2, params_2, _ = ppo.train(
            environment=env_2,
            # INCREASED: Give it enough steps to learn
            num_timesteps=2_000_000,
            num_evals=5,
            restore_params=params_1,  # Inherit weights from Stage 1
            reward_scaling=1.0,
            # INCREASED: Match Stage 1 episode length
            episode_length=500,
            normalize_observations=False, 
            action_repeat=1,
            unroll_length=10,
            num_minibatches=16,       
            num_updates_per_batch=4,
            discounting=0.99,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=128,
            batch_size=80,
            seed=99,
        )
    print("Stage 2 continual training completed.")

    pipeline_end_time = time.time()
    total_time_seconds = pipeline_end_time - pipeline_start_time
    minutes = int(total_time_seconds // 60)
    seconds = int(total_time_seconds % 60)
    
    print("\n=======================================================")
    print(f"TOTAL INFERENCE PIPELINE TIME: {minutes} minutes and {seconds} seconds")
    print("=======================================================")

    # --- STAGE 3: Save model ---
    print("\n[Wrap Up] ---------------------------------------------")
    model_path = "uav_continual_ppo_policy.pkl"
    brax_model.save_params(model_path, params_2)
    print(f"Final model saved to: '{model_path}'")

    # --- STAGE 4: Inference ---
    print("\nStarting UAV flight test on Batch 2 trajectories...")
    loaded_params = brax_model.load_params(model_path)

    # Standalone Inference Reconstruction
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo_train

    ppo_network = ppo_networks.make_ppo_networks(
        env_2.observation_size,
        env_2.action_size,
        preprocess_observations_fn=lambda x, y: x  
    )

    standalone_inference_generator = ppo_networks.make_inference_fn(ppo_network)
    policy_fn = standalone_inference_generator(loaded_params)

    jit_reset  = jax.jit(env_2.reset)
    jit_step   = jax.jit(env_2.step)
    jit_policy = jax.jit(policy_fn)

    rng = jax.random.PRNGKey(123)
    rng, rng_reset = jax.random.split(rng)
    state = jit_reset(rng_reset)
    print("UAV Takeoff!")

    rollout = [state.pipeline_state]

    # INCREASED: Allow recording up to 500 frames (5 seconds physics time)
    for step in range(500):
        rng, rng_act = jax.random.split(rng)
        ctrl, _ = jit_policy(state.obs, rng_act)
        state = jit_step(state, ctrl)
        
        rollout.append(state.pipeline_state)

        if step % 50 == 0:
            dist = state.metrics['distance_to_target']
            target_idx = state.metrics['target_idx']
            print(f"Time Step {step:03d} | Tracking Waypoint {int(target_idx)} | Distance Error: {dist:.3f} m")

        if state.done:
            dist = state.metrics['distance_to_target']
            print(f"Episode terminated early at step {step}. Final Distance: {dist:.3f} m")
            break

    print("Inference flight sequence completed!")
    
    print("\n[Visualization] Generating 3D flight trajectory HTML...")
    html_content = html.render(env_2.sys.tree_replace({'opt.timestep': env_2.dt}), rollout)
    html_path = "uav_flight_trajectory.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Visualization saved to '{html_path}'.")

if __name__ == "__main__":
    main()
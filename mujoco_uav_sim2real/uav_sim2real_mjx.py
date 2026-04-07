"""
UAV Sim2Real Training Demo with MuJoCo XLA (MJX), JAX, Brax using OpenSource Dataset from Hugging Face.
Optimized drone reinforcement learning training script for TPU v5e-8 / Kaggle TPU v3-8.
"""

import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx
from brax import envs
from brax.envs.base import State, Env
from brax.training.agents.ppo import train as ppo
from brax.io import model as brax_model  # For saving and loading models
from datasets import load_dataset, Dataset
import numpy as np
import ast
import time

# ==========================================
# 1. Hugging Face Dataset Integration 
# ==========================================
def load_hf_trajectory_dataset(num_points=10): # Reduced to 10 points for a quick demo
    """
    Load real-world UAV flight trajectory data from Hugging Face Hub.
    Using open-source dataset: riotu-lab/Synthetic-UAV-Flight-Trajectories
    """
    print(f"Downloading {num_points} waypoints from Hugging Face...")
    
    try:
        dataset = load_dataset("riotu-lab/Synthetic-UAV-Flight-Trajectories", split="train")
        subset = dataset.select(range(num_points))
        
        if 'x' in subset.column_names and 'y' in subset.column_names and 'z' in subset.column_names:
            waypoints = np.column_stack((subset['x'], subset['y'], subset['z']))
        elif 'position' in subset.column_names:
            positions = subset['position']
            if isinstance(positions[0], str):
                positions = [ast.literal_eval(p) for p in positions]
            waypoints = np.array(positions)
        else:
            raise ValueError("Columns not recognized")
            
        print(f"Successfully loaded {len(waypoints)} waypoints!")
        
    except Exception as e:
        print(f"Could not load HF dataset online ({e}). Using local fallback trajectory...")
        waypoints = np.array([
            [1.0, 0.0, 2.0], [2.0, 1.0, 2.5], [2.0, -1.0, 2.0], [0.0, 0.0, 1.5],
            [1.0, 1.0, 1.5], [2.0, 2.0, 2.0], [3.0, 1.0, 2.5], [3.0, 0.0, 2.0],
            [2.0, -1.0, 1.5], [1.0, 0.0, 1.0]
        ])

    return jnp.array(waypoints)

# ==========================================
# 2. MuJoCo UAV (Quadrotor) Model Definition (XML)
# ==========================================
UAV_XML = """
<mujoco model="quadrotor">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option gravity="0 0 -9.81" timestep="0.01" integrator="RK4"/>
  
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
class UAVTrackingEnv(Env):
    def __init__(self, waypoints):
        super().__init__()
        self.sys = mujoco.MjModel.from_xml_string(UAV_XML)
        self.sys_mjx = mjx.put_model(self.sys)
        self.waypoints = waypoints
        self.num_waypoints = waypoints.shape[0]

    def reset(self, rng: jnp.ndarray) -> State:
        rng, rng_state = jax.random.split(rng)
        data = mjx.make_data(self.sys_mjx)
        
        # Initial position perturbation
        qpos = self.sys_mjx.qpos0 + jax.random.uniform(rng_state, (self.sys.nq,), minval=-0.1, maxval=0.1)
        data = data.replace(qpos=qpos)
        
        data = mjx.forward(self.sys_mjx, data)
        obs = self._get_obs(data, target_idx=0)
        
        return State(
            pipeline_state=data, obs=obs, reward=jnp.array(0.0), done=jnp.array(0.0),
            metrics={"target_idx": jnp.array(0), "distance_to_target": jnp.array(0.0)}
        )

    def step(self, state: State, action: jnp.ndarray) -> State:
        data = state.pipeline_state
        target_idx = state.metrics["target_idx"]
        
        data = data.replace(ctrl=action)
        data = mjx.step(self.sys_mjx, data)
        
        uav_pos = data.qpos[:3]
        target_pos = self.waypoints[target_idx]
        distance = jnp.linalg.norm(uav_pos - target_pos)
        
        reward = -distance 
        
        reached = distance < 0.2
        target_idx = jnp.where(reached, jnp.minimum(target_idx + 1, self.num_waypoints - 1), target_idx)
        
        done = jnp.where(uav_pos[2] < 0.1, 1.0, 0.0) 
        done = jnp.where(distance > 5.0, 1.0, done)
        
        obs = self._get_obs(data, target_idx)
        metrics = {"target_idx": target_idx, "distance_to_target": distance}
        
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics)

    def _get_obs(self, data: mjx.Data, target_idx: int) -> jnp.ndarray:
        uav_pos, uav_quat, uav_vel = data.qpos[:3], data.qpos[3:7], data.qvel[:3]
        target_pos = self.waypoints[target_idx]
        return jnp.concatenate([uav_pos, uav_quat, uav_vel, target_pos])

    @property
    def action_size(self): return 4
    @property
    def observation_size(self): return 13

# ==========================================
# 4. Main Flow: Train -> Save -> Load -> Inference
# ==========================================
def main():
    print(f"JAX Devices: {jax.device_count()} (TPU cores expected: 8 on Kaggle)")
    
    # --- PHASE 1: PREPARE ENV ---
    waypoints = load_hf_trajectory_dataset(num_points=10) # 10 points for quick demo
    env = UAVTrackingEnv(waypoints)
    env_fn = lambda: env
    
    # --- PHASE 2: TRAINING ---
    print("\n--- Starting PPO Training ---")
    start_time = time.time()
    
    # Reduced num_timesteps to 100,000 so it finishes in a few minutes on Kaggle
    make_inference_fn, params, _ = ppo.train(
        environment=env_fn,
        num_timesteps=100_000,   
        num_evals=8,
        reward_scaling=1.0,
        episode_length=200,      
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,        
        num_updates_per_batch=4,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,           # Utilize TPU parallelization
        batch_size=1024,
        seed=42,
    )
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")

    # --- PHASE 3: SAVING MODEL ---
    print("\n--- Saving Model ---")
    model_path = "uav_ppo_policy.pkl"
    brax_model.save(params, model_path)
    print(f"Model successfully saved to '{model_path}'.")

    # --- PHASE 4: INFERENCE (TESTING THE LOADED MODEL) ---
    print("\n--- Running Inference with Saved Model ---")
    
    # 1. Load params from disk
    loaded_params = brax_model.load(model_path)
    print("Model loaded from disk.")
    
    # 2. Create the policy function using the loaded params
    # (make_inference_fn handles the observation normalization internally)
    policy_fn = make_inference_fn(loaded_params)
    
    # 3. JIT compile the env functions and policy for fast execution
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_policy = jax.jit(policy_fn)
    
    # 4. Run a single test episode
    rng = jax.random.PRNGKey(123)
    rng, rng_reset = jax.random.split(rng)
    
    state = jit_reset(rng_reset)
    print("Takeoff!")
    
    for step in range(200):
        rng, rng_act = jax.random.split(rng)
        
        # Get action from the loaded policy
        ctrl, _ = jit_policy(state.obs, rng_act)
        
        # Step the environment
        state = jit_step(state, ctrl)
        
        if step % 20 == 0:
            dist = state.metrics['distance_to_target']
            target_idx = state.metrics['target_idx']
            print(f"Step {step:03d} | Tracking Waypoint {target_idx} | Distance Error: {dist:.3f}m")
            
        if state.done:
            print(f"Episode terminated early at step {step} (Crashed or went out of bounds).")
            break
            
    print("Inference Demo Complete!")

if __name__ == "__main__":
    main()
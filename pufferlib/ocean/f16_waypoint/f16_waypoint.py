"""F16 Waypoint Environment - PufferLib-compatible vectorized environment

All simulation logic is in f16_waypoint.h (calls f16_model.h and raylib_renderer.h).
This is a thin PufferLib wrapper over the C binding.
"""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.f16_waypoint import binding


class F16Waypoint(pufferlib.PufferEnv):
    """F16 Waypoint navigation environment with vectorization support"""
    
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, 
                 step_size=1/30, time_limit=100.0, buf=None, seed=0):
        """Initialize F16 Waypoint environment
        
        Args:
            num_envs: Number of parallel environments
            render_mode: Rendering mode (unused, kept for compatibility)
            log_interval: Steps between logging episode statistics
            step_size: Simulation timestep in seconds (default: 1/30)
            time_limit: Episode time limit in seconds (default: 100.0)
            buf: Optional pre-allocated buffer (PufferLib internal)
            seed: Random seed
        """
        # Observation: 28D continuous (see OBSERVATION_SPACE.md)
        # [0-18] F16 state (sin/cos angles, velocities, rates, altitudes)
        # [19-21] Previous actions (Nz, ps, Ny_r)
        # [22-27] Waypoint in spherical coordinates
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )
        
        # Action: 4D continuous
        # [0] Nz: Normal acceleration command (G's)
        # [1] ps: Roll rate command (rad/s)
        # [2] Ny_r: Lateral acceleration command (G's)
        # [3] throttle: Engine throttle (0-1)
        self.single_action_space = gymnasium.spaces.Box(
            low=np.array([-2.0, -3.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([6.0, 3.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        
        # Initialize PufferEnv base class (allocates buffers)
        super().__init__(buf)
        
        # Initialize C environments
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            step_size=step_size, time_limit=time_limit
        )
 
    def reset(self, seed=0):
        """Reset all environments
        
        Args:
            seed: Random seed for reset
            
        Returns:
            observations: Initial observations for all environments
            infos: Empty list (populated during steps)
        """
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        """Step all environments
        
        Args:
            actions: Action array of shape (num_envs, 4)
            
        Returns:
            observations: Next observations
            rewards: Rewards for each environment
            terminals: Terminal flags
            truncations: Truncation flags
            infos: List of info dicts (includes logs at log_interval)
        """
        self.tick += 1

        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        """Render environment 0 (for visualization during training)"""
        binding.vec_render(self.c_envs, 0)

    def close(self):
        """Close all environments and free resources"""
        binding.vec_close(self.c_envs)


if __name__ == '__main__':
    """Speed test for vectorized F16 Waypoint environment"""
    import time
    
    N = 1024  # Number of parallel environments
    
    print("F16 Waypoint Vectorized Speed Test")
    print("=" * 50)
    print(f"Environments: {N}")
    
    env = F16Waypoint(num_envs=N, step_size=1/30, time_limit=100.0, seed=42)
    env.reset()
    
    steps = 0
    
    # Pre-generate random actions for consistent testing
    CACHE = 1024
    actions = np.random.uniform(
        low=[-2.0, -3.0, -1.0, 0.0],
        high=[6.0, 3.0, 1.0, 1.0],
        size=(CACHE, N, 4)
    ).astype(np.float32)
    
    i = 0
    start = time.time()
    print("Running for 10 seconds...")
    
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1
    
    elapsed = time.time() - start
    sps = int(steps / elapsed)
    
    print(f"Total steps: {steps:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"F16 Waypoint SPS: {sps:,}")
    print("=" * 50)
    
    env.close()

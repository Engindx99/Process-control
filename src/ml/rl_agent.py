import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from control.mpc import AdvancedRotaryKilnMPC
from physics.digital_twin import AdvancedRotaryKiln


class RotaryKilnEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # ---------------- SYSTEM ----------------
        self.mpc = AdvancedRotaryKilnMPC()
        self.plant = AdvancedRotaryKiln()

        # ---------------- RL SPACE ----------------
        self.action_space = gym.spaces.Box(
            low=-10, high=10, shape=(1,), dtype=np.float32
        )

        # state: [temp, oxygen]
        self.observation_space = gym.spaces.Box(
            low=np.array([800, 0]),
            high=np.array([1600, 10]),
            dtype=np.float32
        )

        # ---------------- INITIALS ----------------
        self.base_setpoint = 1400.0
        self.setpoint = 1400.0
        self.feed_rate = 5.0

    def step(self, action):

        # 1) RL SETPOINT
        self.setpoint = self.base_setpoint + float(action[0])

        # 2) MPC update (TVP injection)
        def tvp_fun(t_now):
            tvp = self.mpc.mpc.get_tvp_template()
            tvp['_tvp', :, 'temp_setpoint'] = self.setpoint
            tvp['_tvp', :, 'material_feed'] = self.feed_rate
            return tvp

        self.mpc.mpc.set_tvp_fun(tvp_fun)

        # 3) MPC CONTROL
        temp = self.plant.current_temp
        fuel, fan = self.mpc.step(temp)

        # 4) REAL DIGITAL TWIN DYNAMICS
        new_temp = self.plant.calculate_dynamics(fuel, self.feed_rate, fan)
        new_o2 = self.plant.oxygen_level

        # 5) REWARD
        error = abs(new_temp - self.setpoint)
        energy_penalty = 0.01 * fuel + 0.001 * fan

        reward = -error - energy_penalty

        # 6) UPDATE STATE
        self.feed_rate = np.clip(
            self.feed_rate + np.random.normal(0, 0.1),
            3.0, 7.0
        )

        obs = np.array([new_temp, new_o2], dtype=np.float32)

        return obs, reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.plant.reset()
        self.setpoint = self.base_setpoint
        self.feed_rate = 5.0

        obs = np.array(
            [self.plant.current_temp, self.plant.oxygen_level],
            dtype=np.float32
        )

        return obs, {}


# ---------------- TRAIN ----------------
if __name__ == "__main__":

    env = RotaryKilnEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99
    )

    print("Training started: RL + MPC + Digital Twin")

    model.learn(total_timesteps=20000)

    model.save("kiln_hybrid_rl_mpc")
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.digital_twin.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC


# =========================
# ENVIRONMENT
# =========================
class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Plant + MPC
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(
            prediction_horizon=cfg['mpc']['prediction_horizon'],
            control_horizon=cfg['mpc']['control_horizon']
        )

        self.setpoint = cfg['mpc']['setpoint']

        # =========================
        # ACTION SPACE (NORMALIZED)
        # =========================
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # =========================
        # OBS SPACE
        # =========================
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        self.prev_temp = 1400.0
        self.last_action = 0.0

        # RL influence scale
        self.action_scale = cfg['rl']['action_limit']

        self.step_count = 0
        self.max_steps = 2000

    # =========================
    # OBSERVATION NORMALIZATION
    # =========================
    def _get_obs(self):
        return np.array([
            (self.plant.temp - self.setpoint) / 100.0,   # normalized error
            (self.plant.temp - self.prev_temp) / 10.0,   # temp change
            self.plant.o2 / 10.0,                        # normalized oxygen
            self.plant.fuel / 20.0                       # normalized fuel
        ], dtype=np.float32)

    # =========================
    # STEP
    # =========================
    def step(self, action):
        self.step_count += 1

        # MPC action
        u_mpc = self.mpc.optimize(self.plant)

        # RL residual (scaled)
        rl_action = np.tanh(action[0]) * 0.05

        # Total control
        total_fuel = u_mpc + rl_action

        # Save previous state
        self.prev_temp = self.plant.temp

        # Plant step
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)

        # =========================
        # REWARD FUNCTION
        # =========================
        error = temp - self.setpoint

        reward = -(error ** 2 * 0.1)

        # smooth control penalty
        action_diff = abs(action[0] - self.last_action)
        reward -= action_diff * 2.0

        # small stability bonus
        if abs(error) < 1.0:
            reward += 2.0

        self.last_action = float(action[0])

        # =========================
        # TERMINATION LOGIC
        # =========================
        terminated = abs(error) < 0.5
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    # =========================
    # RESET
    # =========================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(
            prediction_horizon=self.cfg['mpc']['prediction_horizon'],
            control_horizon=self.cfg['mpc']['control_horizon']
        )

        self.prev_temp = 1400.0
        self.last_action = 0.0
        self.step_count = 0

        return self._get_obs(), {}


# =========================
# ENV FACTORY
# =========================
def make_env(cfg):
    def _init():
        return ResidualKilnEnv(cfg)
    return _init


# =========================
# TRAIN FUNCTION
# =========================
def train_model(cfg):

    env = SubprocVecEnv(
        [make_env(cfg) for _ in range(cfg['hardware']['num_cpu'])]
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=float(cfg['rl']['learning_rate']),
        n_steps=cfg['rl']['n_steps'],
        batch_size=cfg['rl']['batch_size'],
        gamma=cfg['rl']['gamma'],
        device=cfg['hardware']['device']
    )

    model.learn(total_timesteps=cfg['rl']['total_timesteps'])

    return model


# =========================
# TEST FUNCTION
# =========================
def run_test(model, cfg):
    env = ResidualKilnEnv(cfg)
    obs, _ = env.reset()

    history = []

    for i in range(5000):

        # --- MPC ---
        u_mpc = env.mpc.optimize(env.plant)

        # --- RL ---
        action, _ = model.predict(obs, deterministic=True)

        # --- STEP ---
        obs, _, _, _, _ = env.step(action)

        # =========================
        # FIX: MPC LOG EKLENDİ
        # =========================
        env.mpc.log_step(
            step=i,
            fuel=float(u_mpc),
            temp=float(env.plant.temp),
            o2=float(env.plant.o2)
        )

        # --- CSV HISTORY ---
        history.append({
            "step": i,
            "temp": float(env.plant.temp),
            "u_rl": float(action[0]),
            "u_mpc": float(u_mpc),
            "fuel": float(env.plant.fuel)
        })

    return history, env
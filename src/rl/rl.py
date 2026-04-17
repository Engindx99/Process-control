import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import yaml
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.dt.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

# =========================================================
# LOGGER
# =========================================================
logger = logging.getLogger("RL_TRAINER")
logger.setLevel(logging.INFO)

if len(logger.handlers) == 0:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =========================================================
# ENVIRONMENT
# =========================================================
class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.cfg = cfg
        self.debug = debug

        # ---------------- SYSTEM ----------------
        self.setpoint = cfg["system"]["setpoint"]
        self.temp_min = cfg["plant"]["temp_min"]
        self.temp_max = cfg["plant"]["temp_max"]
        self.max_steps = cfg.get("env", {}).get("max_steps", 3000)

        # ---------------- OBS ----------------
        self.temp_scale = cfg["observation"]["temp_scale"]
        self.o2_scale = cfg["observation"]["o2_scale"]
        self.fuel_scale = cfg["observation"]["fuel_scale"]

        # ---------------- ACTION ----------------
        self.action_limit = cfg["rl"]["action_limit"]

        # ---------------- ENV ----------------
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(cfg)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self.reset()

    def _get_obs(self, eff=0.8):
        temp_error = self.plant.temp - self.setpoint
        base_obs = [
            temp_error / self.temp_scale,
            (self.plant.temp - self.prev_temp) / self.temp_scale,
            (self.plant.o2 - 3.0) / self.o2_scale,
            eff
        ]
        fuel_memory = np.array(self.plant.fuel_history[-10:], dtype=np.float32) / self.fuel_scale
        return np.concatenate([base_obs, fuel_memory]).astype(np.float32)

    def step(self, action):
        try:
            # 1. MPC Baseline
            u_mpc_f, u_mpc_v = self.mpc.optimize(self.plant)

            # 2. RL Residual
            res_f = action[0] * self.action_limit
            res_v = action[1] * self.action_limit * 50.0

            self.prev_temp = self.plant.temp
            current_fuel = u_mpc_f + res_f
            current_fan = u_mpc_v + res_v

            # 3. DT Step
            result = self.plant.step(current_fuel, current_fan)
            
            temp = result["Temperature"]
            o2 = result["O2"]
            eff = result["Efficiency"]

            # 4. Reward Calculation (DYNAMIC FROM CONFIG)
            rew_cfg = self.cfg["reward"]
            temp_err = abs(temp - self.setpoint)
            
            # Temperature Reward
            if temp_err > rew_cfg.get("temp_deadband", 10.0):
                r_temp = -(temp_err ** 2) * rew_cfg.get("big_error_scale", 0.005)
            else:
                r_temp = -temp_err * rew_cfg.get("small_error_scale", 0.05)

            # Stability & Smoothing Rewards
            r_damping = -abs(temp - self.prev_temp) * rew_cfg.get("damping_scale", 0.2)
            r_smooth = -np.mean(np.square(action)) * rew_cfg.get("smooth_scale", 2.0)
            r_energy = -current_fuel * rew_cfg.get("energy_scale", 0.02)

            # Bonus Logic
            bonus = 0.0
            if temp_err < rew_cfg.get("temp_deadband", 10.0):
                bonus += rew_cfg.get("bonus_small", 1.0)
                if temp_err < 2.0:
                    # Kritik bonusu kontrol altında tutmak için temp_err + 0.1 bölmesi korunur
                    bonus += rew_cfg.get("bonus_critical", 5.0) / (temp_err + 0.1)

            reward = r_temp + r_damping + r_smooth + r_energy + bonus + (eff * 0.5)

            # 5. Termination & Truncation
            terminated = (temp < self.temp_min or temp > self.temp_max)
            truncated = self.step_count >= self.max_steps

            if terminated:
                reward -= 100.0 # Daha sert bir ceza (opsiyonel)

            self.step_count += 1
            return self._get_obs(eff), float(reward), terminated, truncated, {}

        except Exception as e:
            logger.error(f"Env error: {e}")
            return self._get_obs(), -100.0, True, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        if seed is not None:
            np.random.seed(seed)
            self.plant.temp = np.random.uniform(self.setpoint - 40, self.setpoint + 40)
        self.prev_temp = self.plant.temp
        self.step_count = 0
        return self._get_obs(), {}

# =========================================================
# UTILS & TRAINING
# =========================================================
def make_env(cfg, rank=0, seed=0):
    def _init():
        env = ResidualKilnEnv(cfg)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_rl(cfg):
    num_cpu = cfg["hardware"].get("num_cpu", 4)
    env = SubprocVecEnv([make_env(cfg, i) for i in range(num_cpu)])
    env = VecMonitor(env)

    model_path = cfg["paths"]["model_save_path"]

    # Sıfırdan başlamak için mevcut .zip dosyasını silmeli veya adını değiştirmelisin!
    if os.path.exists(model_path + ".zip"):
        logger.info(f"Loading model for fine-tuning: {model_path}")
        model = PPO.load(model_path, env=env)
        model.learning_rate = float(cfg["rl"]["learning_rate"])
    else:
        logger.info("Starting fresh training (New configuration)")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=float(cfg["rl"]["learning_rate"]),
            n_steps=cfg["rl"]["n_steps"],
            batch_size=cfg["rl"]["batch_size"],
            gamma=cfg["rl"]["gamma"],
            device=cfg["hardware"]["device"]
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 40000 // num_cpu),
        save_path="./models/checkpoints/",
        name_prefix="kiln_rl_fresh"
    )

    model.learn(total_timesteps=cfg["rl"]["total_timesteps"], callback=checkpoint_callback)
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    logger.info("Training session completed.")
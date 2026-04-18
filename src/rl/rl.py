import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.dt.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

logger = logging.getLogger("RL_TRAINER")

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=1450.0):
        self.q = process_variance      
        self.r = measurement_variance  
        self.x = initial_value         
        self.p = 1.0                   

    def update(self, measurement):
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x

class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setpoint = cfg["system"]["setpoint"]
        self.temp_min = cfg["plant"]["temp_min"]
        self.temp_max = cfg["plant"]["temp_max"]
        self.action_limit = cfg["rl"]["action_limit"]
        self.max_steps = 3000

        self.temp_scale = cfg["observation"]["temp_scale"]
        self.o2_scale = cfg["observation"]["o2_scale"]
        self.fuel_scale = cfg["observation"]["fuel_scale"]

        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(cfg)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.reset()

    def get_obs(self, filtered_temp, eff=0.8):
        temp_error = (filtered_temp - self.setpoint) / self.temp_scale
        base_obs = [
            temp_error,
            (filtered_temp - self.prev_filtered_temp) / self.temp_scale,
            (self.plant.o2 - 3.0) / self.o2_scale,
            eff,
            self.prev_raw_action[0],
            self.prev_raw_action[1]
        ]
        fuel_memory = np.array(self.plant.fuel_history[-8:], dtype=np.float32) / self.fuel_scale
        return np.concatenate([base_obs, fuel_memory]).astype(np.float32)

    def step(self, action):
        # 1. Sensör Filtreleme
        filtered_temp = self.kf_temp.update(self.plant.temp)

        # 2. MPC ve RL Katkısı
        u_mpc_f, u_mpc_v = self.mpc.optimize(self.plant)
        res_f = action[0] * self.action_limit
        res_v = action[1] * self.action_limit * 50.0

        target_fuel = max(0, u_mpc_f + res_f)
        target_fan = max(0, u_mpc_v + res_v)

        # --- KRİTİK: AKSİYON SÖNÜMLEME (AMORTİSÖR) ---
        # Ajan ne kadar zıplarsa zıplasın, fiziksel vana %80 eski konumunu korur.
        # Bu işlem v7 ve v8'deki o testere dişlerini yok eder.
        current_fuel = (0.8 * self.prev_applied_fuel) + (0.2 * target_fuel)
        current_fan = (0.8 * self.prev_applied_fan) + (0.2 * target_fan)

        # 3. Simülasyon
        result = self.plant.step(current_fuel, current_fan)
        new_raw_temp = result["Temperature"]
        eff = result["Efficiency"]

        # 4. Ödül Hesaplama
        rew_cfg = self.cfg["reward"]
        temp_err = abs(filtered_temp - self.setpoint)
        
        if temp_err > rew_cfg.get("temp_deadband", 0.8):
            r_temp = -(temp_err ** 2) * rew_cfg.get("big_error_scale", 0.5)
        else:
            r_temp = (-temp_err * rew_cfg.get("small_error_scale", 2.0)) + rew_cfg.get("bonus_small", 10.0)

        # Jitter ve Smoothness cezası artık sönümlenmiş aksiyon üzerinden
        delta_action = np.abs(action - self.prev_raw_action)
        r_jitter = -np.mean(delta_action) * rew_cfg.get("jitter_penalty", 60.0)
        
        reward = r_temp + r_jitter + (eff * 0.5)

        # 5. Kayıtlar
        self.prev_filtered_temp = filtered_temp
        self.prev_raw_action = action.copy()
        self.prev_applied_fuel = current_fuel
        self.prev_applied_fan = current_fan
        self.step_count += 1

        terminated = (new_raw_temp < self.temp_min or new_raw_temp > self.temp_max)
        truncated = self.step_count >= self.max_steps
        if terminated: reward -= 100.0

        return self.get_obs(filtered_temp, eff), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin(seed=seed)
        
        # Kalman Otoritesi
        mv = 4.0 
        self.kf_temp = KalmanFilter(process_variance=0.001, measurement_variance=mv, initial_value=self.plant.temp)
        
        self.prev_filtered_temp = self.plant.temp
        self.prev_raw_action = np.zeros(2, dtype=np.float32)
        
        # Sönümleme için başlangıç değerleri
        self.prev_applied_fuel = 15.0 # Ortalama yakıt
        self.prev_applied_fan = 1000.0 # Ortalama fan
        
        self.step_count = 0
        obs_temp = self.kf_temp.update(self.plant.temp)
        return self.get_obs(obs_temp), {}

def train_rl(cfg):
    num_cpu = cfg["hardware"].get("num_cpu", 4)
    env = SubprocVecEnv([lambda i=i: ResidualKilnEnv(cfg) for i in range(num_cpu)])
    env = VecMonitor(env)

    model_path = cfg["paths"]["model_save_path"]
    
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

    model.learn(total_timesteps=cfg["rl"]["total_timesteps"])
    model.save(model_path)
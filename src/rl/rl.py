import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import logging
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Proje içi importlar
from src.dt.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

logger = logging.getLogger("RL_ENVIRONMENT")

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
        self.max_steps = 360  
        
        # Konfigürasyon ve Limitler (Config'den çekiliyor)
        self.temp_min = cfg["plant"]["temp_min"]
        self.temp_max = cfg["plant"]["temp_max"]
        self.action_limit = cfg["rl"].get("action_limit", 0.5)
        self.fan_action_limit = cfg["rl"].get("fan_action_limit", 10.0)
        
        self.temp_scale = cfg["observation"]["temp_scale"]
        self.o2_scale = cfg["observation"]["o2_scale"]
        self.fuel_scale = cfg["observation"]["fuel_scale"]

        self.plant = RotaryKilnDigitalTwin(config=cfg)
        self.mpc = MPC(cfg)
        self.fuel_memory = deque(maxlen=8)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

    def get_obs(self, filtered_temp, current_target, eff):
        temp_error = (filtered_temp - current_target) / self.temp_scale
        delta_temp = (filtered_temp - self.prev_filtered_temp) / self.temp_scale
        
        while len(self.fuel_memory) < 8:
            self.fuel_memory.append(self.plant.fuel / self.fuel_scale)

        base_obs = [
            temp_error,
            delta_temp,
            (self.plant.o2 - 3.2) / self.o2_scale,
            eff,
            self.prev_raw_action[0],
            self.prev_raw_action[1]
        ]
        
        fuel_hist_list = list(self.fuel_memory)
        return np.concatenate([base_obs, fuel_hist_list]).astype(np.float32)

    def step(self, action):
        current_target = self.plant.get_staircase_target(self.step_count)
        
        # 1. MPC + RL Residual Birleşimi (Limitler dinamik)
        u_mpc_f, u_mpc_v = self.mpc.optimize(self.plant, current_target=current_target)
        
        res_f = action[0] * self.action_limit
        res_v = action[1] * self.fan_action_limit

        target_fuel = u_mpc_f + res_f
        target_fan = u_mpc_v + res_v

        # 2. Simülasyon Adımı
        result = self.plant.step(target_fuel, target_fan)
        
        new_temp = result.get("temperature") or result.get("Temperature")
        new_fuel = result.get("fuel") or result.get("Fuel")
        eff = result.get("efficiency") or result.get("Efficiency") or 0.8
        
        self.fuel_memory.append(new_fuel / self.fuel_scale)
        filtered_temp = self.kf_temp.update(new_temp)

        # 3. Dinamik Ödül (Config katsayıları ile)
        temp_err = abs(filtered_temp - current_target)
        
        # Sıcaklık cezası
        r_temp = -(temp_err ** 2) * self.cfg["reward"].get("error_penalty_scale", 15.0)
        
        # Yakınlık bonusu
        if temp_err < 0.5:
            r_temp += 5.0 / (temp_err + 0.1)

        # Aksiyon pürüzsüzlüğü
        r_smooth = -np.sum((action - self.prev_raw_action)**2) * self.cfg["reward"].get("smoothness_weight", 20.0)
        
        # Verimlilik ödülü
        r_eff = eff * self.cfg["reward"].get("efficiency_weight", 3.0)
        
        reward = r_temp + r_smooth + r_eff

        # 4. State Güncelleme
        self.prev_filtered_temp = filtered_temp
        self.prev_raw_action = action.copy()
        self.step_count += 1

        # 5. Terminal Kontrolleri
        terminated = (new_temp < self.temp_min or new_temp > self.temp_max)
        truncated = self.step_count >= self.max_steps
        
        if terminated: 
            reward -= self.cfg["reward"].get("termination_penalty", 1000.0)

        return self.get_obs(filtered_temp, current_target, eff), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant.reset()
        self.fuel_memory.clear()
        for _ in range(8):
            self.fuel_memory.append(self.plant.fuel / self.fuel_scale)
        
        self.kf_temp = KalmanFilter(0.001, 4.0, initial_value=self.plant.temp)
        self.prev_filtered_temp = self.plant.temp
        self.prev_raw_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        
        current_target = self.plant.get_staircase_target(0)
        return self.get_obs(self.plant.temp, current_target, 0.8), {}

def train_rl(cfg):
    num_cpu = cfg["hardware"].get("num_cpu", 11)
    env = SubprocVecEnv([lambda i=i: ResidualKilnEnv(cfg) for i in range(num_cpu)])
    env = VecMonitor(env)

    model_path = cfg["paths"]["model_save_path"]
    
    if os.path.exists(model_path + ".zip"):
        print(f"--- [Fine-Tune] Mevcut model yükleniyor ---")
        model = PPO.load(model_path, env=env, device=cfg["hardware"]["device"])
    else:
        print("--- [Sıfırdan Eğitim] Yeni model oluşturuluyor ---")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=float(cfg["rl"].get("learning_rate", 3e-4)),
            n_steps=cfg["rl"].get("n_steps", 2048),
            batch_size=cfg["rl"].get("batch_size", 128),
            gamma=cfg["rl"].get("gamma", 0.99),
            device=cfg["hardware"]["device"]
        )

    model.learn(total_timesteps=cfg["rl"]["total_timesteps"], progress_bar=True)
    model.save(model_path)
    print(f"--- Model kaydedildi: {model_path} ---")
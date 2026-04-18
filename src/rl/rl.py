import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from src.dt.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC
import os

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
        # 1. Filtreleme
        filtered_temp = self.kf_temp.update(self.plant.temp)

        # 2. Kontrolör Katkıları
        u_mpc_f, u_mpc_v = self.mpc.optimize(self.plant)
        res_f = action[0] * self.action_limit
        res_v = action[1] * self.action_limit * 50.0

        target_fuel = max(0, u_mpc_f + res_f)
        target_fan = max(0, u_mpc_v + res_v)

        # Amortisör (%50 Çeviklik)
        current_fuel = (0.5 * self.prev_applied_fuel) + (0.5 * target_fuel)
        current_fan = (0.5 * self.prev_applied_fan) + (0.5 * target_fan)

        # 3. Fiziksel Simülasyon
        result = self.plant.step(current_fuel, current_fan)
        new_raw_temp = result["Temperature"]
        eff = result["Efficiency"]

        # 4. Dinamik Ödül Hesaplama
        rew_cfg = self.cfg["reward"]
        temp_err = abs(filtered_temp - self.setpoint)
        
        # Sıcaklık Ödülü (Hassas Bölge Regülasyonlu)
        if temp_err > rew_cfg.get("temp_deadband", 0.2):
            r_temp = -(temp_err ** 2) * rew_cfg.get("big_error_scale", 2.0)
        else:
            # Burası MAE'yi tek haneye indirecek olan motor:
            r_temp = (-temp_err * rew_cfg.get("small_error_scale", 5.8)) + rew_cfg.get("bonus_small", 10.0)

        # Pürüzsüzlük Cezaları (Config üzerinden 60.0 ve 15.0)
        delta_action = np.abs(action - self.prev_raw_action)
        r_jitter = -np.mean(delta_action) * rew_cfg.get("jitter_penalty", 60.0)
        r_smooth = -np.sum(delta_action**2) * rew_cfg.get("smooth_scale", 15.0)
        
        reward = r_temp + r_jitter + r_smooth + (eff * 0.5)

        # 5. Kayıt ve Terminal Kontrolü
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
        mv = 4.0 
        self.kf_temp = KalmanFilter(process_variance=0.001, measurement_variance=mv, initial_value=self.plant.temp)
        self.prev_filtered_temp = self.plant.temp
        self.prev_raw_action = np.zeros(2, dtype=np.float32)
        self.prev_applied_fuel = 15.0 
        self.prev_applied_fan = 1000.0 
        self.step_count = 0
        return self.get_obs(self.kf_temp.update(self.plant.temp)), {}

def train_rl(cfg):
    num_cpu = cfg["hardware"].get("num_cpu", 11)
    env = SubprocVecEnv([lambda i=i: ResidualKilnEnv(cfg) for i in range(num_cpu)])
    env = VecMonitor(env)

    model_path = cfg["paths"]["model_save_path"]
    
    # FINE-TUNE MODU: Eğer dosya varsa üzerine yaz, yoksa sıfırdan oluştur
    if os.path.exists(model_path):
        print(f"--- [Fine-Tune] Mevcut model yükleniyor: {model_path} ---")
        model = PPO.load(
            model_path, 
            env=env, 
            device=cfg["hardware"]["device"],
            custom_objects={"learning_rate": float(cfg["rl"]["learning_rate"])}
        )
    else:
        print("--- [Sıfırdan Eğitim] Yeni model oluşturuluyor ---")
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

    model.learn(total_timesteps=cfg["rl"]["total_timesteps"], progress_bar=True)
    model.save(model_path)
    print(f"--- Model başarıyla kaydedildi: {model_path} ---")
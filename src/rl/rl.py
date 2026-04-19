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

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import os

# Kendi yazdığın sınıfları buradan import ettiğini varsayıyoruz
# from src.dt.dt import RotaryKilnDigitalTwin
# from src.mpc.mpc import MPC

class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.max_steps = 360  # 6 saatlik merdiven senaryosu
        
        # Fiziksel Sınırlar ve Scalerlar
        self.temp_min = cfg["plant"]["temp_min"]
        self.temp_max = cfg["plant"]["temp_max"]
        self.action_limit = cfg["rl"]["action_limit"]
        self.temp_scale = cfg["observation"]["temp_scale"]
        self.o2_scale = cfg["observation"]["o2_scale"]
        self.fuel_scale = cfg["observation"]["fuel_scale"]

        # Motorlar
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(cfg)

        # Action: [Fuel_Residual, Fan_Residual] -> (-1, 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Observation: [Error, Delta_Temp, O2, Eff, Prev_Action_F, Prev_Action_V, Fuel_Hist(8)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.reset()

    def get_obs(self, filtered_temp, current_target, eff):
        """Ajanın hedefe olan uzaklığını ve fırın durumunu görmesini sağlar."""
        temp_error = (filtered_temp - current_target) / self.temp_scale
        delta_temp = (filtered_temp - self.prev_filtered_temp) / self.temp_scale
        
        base_obs = [
            temp_error,
            delta_temp,
            (self.plant.o2 - 3.2) / self.o2_scale,
            eff,
            self.prev_raw_action[0],
            self.prev_raw_action[1]
        ]
        # Son 8 dakikalık yakıt geçmişi (ataleti anlaması için)
        fuel_memory = np.array(self.plant.fuel_history[-8:], dtype=np.float32) / self.fuel_scale
        return np.concatenate([base_obs, fuel_memory]).astype(np.float32)

    def step(self, action):
        # 1. Merdiven Stratejisinden Güncel Hedefi Al
        current_target = self.plant.get_staircase_target(self.step_count)

        # 2. Sensör Verisini Filtrele (Kalman)
        filtered_temp = self.kf_temp.update(self.plant.temp)

        # 3. MPC Katkısını Al (Dinamik Hedefle)
        u_mpc_f, u_mpc_v = self.mpc.optimize(self.plant, current_target=current_target)
        
        # 4. RL Residual Katkısını Ekle
        res_f = action[0] * self.action_limit
        res_v = action[1] * self.action_limit * 50.0 # Fan için daha geniş skala

        target_fuel = u_mpc_f + res_f
        target_fan = u_mpc_v + res_v

        # 5. Fiziksel Simülasyon Adımı
        result = self.plant.step(target_fuel, target_fan)
        new_raw_temp = result["Temperature"]
        eff = result["Efficiency"]

        # 6. Dinamik Ödül Hesaplama
        rew_cfg = self.cfg["reward"]
        temp_err = abs(filtered_temp - current_target)
        
        # Isı Hatası Ödülü
        if temp_err > 0.2:
            r_temp = -(temp_err ** 2) * rew_cfg.get("big_error_scale", 2.0)
        else:
            # İnce ayar bölgesi (MAE'yi düşüren kısım)
            r_temp = (-temp_err * rew_cfg.get("small_error_scale", 10.0)) + 5.0

        # Aksiyon Pürüzsüzlüğü (Jitter Penalty)
        r_smooth = -np.sum((action - self.prev_raw_action)**2) * 20.0
        
        reward = r_temp + r_smooth + (eff * 2.0)

        # 7. State Güncelleme
        self.prev_filtered_temp = filtered_temp
        self.prev_raw_action = action.copy()
        self.step_count += 1

        # 8. Terminal Kontrolleri
        terminated = (new_raw_temp < self.temp_min or new_raw_temp > self.temp_max)
        truncated = self.step_count >= self.max_steps
        
        if terminated: 
            reward -= 200.0 # Fırın sönerse veya aşırı ısınırsa ağır ceza

        return self.get_obs(filtered_temp, current_target, eff), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin(seed=seed)
        self.plant.reset() # 1300'den başlar
        
        # Kalman başlangıcı
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
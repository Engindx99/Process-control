import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.dt.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(cfg)
        
        # Action Limit: Genelde 0.2 - 0.5 arası idealdir
        self.action_limit = cfg['rl'].get('action_limit', 0.2)
        
        # Action: [-1, 1] arası (0: Yakıt düzeltme, 1: Fan düzeltme)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Obs: [Temp_Error, Temp_Delta, O2_Error, Fuel, Fan, Eff]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def _get_obs(self, eff=0.8):
        # Normalize edilmiş gözlemler RL'in daha hızlı öğrenmesini sağlar
        temp_error = (self.plant.temp - 1450.0) 
        o2_error = (self.plant.o2 - 3.0)
        
        return np.array([
            temp_error / 20.0,       # 20 derecelik hata -> 1.0 sinyali
            (self.plant.temp - self.prev_temp) / 5.0, 
            o2_error / 2.0,
            (self.plant.fuel - 17.5) / 5.0, 
            (self.plant.fan - 1050.0) / 100.0, 
            eff
        ], dtype=np.float32)

    def step(self, action):
        # 1. MPC Kararı (Temel kontrol)
        u_mpc_f, u_mpc_v = self.mpc.optimize(self.plant)
        
        # 2. RL Residual Katkısı (İnce ayar)
        # Fan etkisini biraz daha geniş tutuyoruz (MPC'nin zorlandığı yer)
        res_f = action[0] * self.action_limit
        res_v = action[1] * (self.action_limit * 50) 
        
        self.prev_temp = self.plant.temp
        
        # 3. Fiziksel Adım
        current_fuel = u_mpc_f + res_f
        current_fan = u_mpc_v + res_v
        temp, o2, eff = self.plant.step(current_fuel, current_fan)
        
        # 4. REWARD TASARIMI (KRİTİK KISIM)
        temp_err = abs(temp - 1450.0)
        o2_err = abs(o2 - 3.0)
        
        # A. Sıcaklık Cezası: Karesel (Hata büyüdükçe ceza katlanarak artar)
        # 1442 derecedeki 8 derecelik hata -> -6.4 ceza verir (Eski sistemde -4 idi)
        r_temp = -(temp_err ** 2) * 0.001 
        
        # B. O2 ve Verimlilik (İkincil hedefler)
        r_o2 = -(o2_err ** 2) * 1.0
        r_eff = eff * 2.0 # Verimlilik ödülünü azalttık ki sıcaklığı bozmasın
        
        # C. Stabilite Cezası: Aksiyonların karesini cezalandır
        # Bu, RL'in sürekli zıplamasını (11 sapmayı) engeller
        r_smooth = -np.sum(np.square(action)) * 1.5
        
        # D. Mıknatıs Ödülü: Hedefe çok yakınsa (Örn: < 1.5 derece) büyük bonus
        bonus = 15.0 if temp_err < 1.5 else 0.0
        
        reward = r_temp + r_o2 + r_eff + r_smooth + bonus
        
        self.step_count += 1
        
        # 5. Sonlanma Koşulları
        # Fırın çok soğursa veya çok ısınırsa eğitimi durdur (Ceza ver)
        terminated = (temp < 1350 or temp > 1550 or o2 < 1.5 or o2 > 8.0)
        if terminated: reward -= 100.0
        
        truncated = self.step_count >= 1000 # Daha sık reset daha iyi keşif sağlar
        
        return self._get_obs(eff), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        
        # Resetleme sırasında farklı noktalardan başlamak modelin genelleme yeteneğini artırır
        if seed is not None:
            np.random.seed(seed)
            self.plant.temp = np.random.uniform(1380, 1480)
            self.plant.o2 = np.random.uniform(2.5, 4.5)
        
        self.prev_temp = self.plant.temp
        self.step_count = 0
        return self._get_obs(), {}

# =================================================================
# EĞİTİM YARDIMCI FONKSİYONLARI
# =================================================================

def make_env(cfg, rank, seed=0):
    def _init():
        env = ResidualKilnEnv(cfg)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_rl(cfg):
    num_cpu = cfg['hardware'].get('num_cpu', 11)
    total_timesteps = cfg['rl'].get('total_timesteps', 200000)
    
    # SubprocVecEnv kullanımı FPS'i yukarıda tutar
    env = SubprocVecEnv([make_env(cfg, i) for i in range(num_cpu)])
    env = VecMonitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4, # Hızı düşürdük (daha kararlı öğrenme için)
        n_steps=1024,       # Her güncelleme için adım sayısı
        batch_size=128,
        ent_coef=0.01,      # Biraz keşif (exploration) ekledik
        verbose=1,
        device=cfg['hardware'].get('device', 'auto')
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000 // num_cpu, 
        save_path='./models/checkpoints/',
        name_prefix='hybrid_kiln'
    )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(cfg['paths'].get('model_save_path', "models/ppo_kiln_v2"))
    print("✅ Eğitim Tamamlandı!")
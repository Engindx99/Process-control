import os
import yaml
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import multiprocessing

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Kendi modüllerin (src klasöründeki yolların doğruluğundan emin ol)
from src.digital_twin.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

# =================================================================
# 1. ORTAM TANIMI (ENVIRONMENT)
# =================================================================
class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super(ResidualKilnEnv, self).__init__()
        self.cfg = cfg
        
        # Fiziksel model ve ana kontrolcü
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(
            prediction_horizon=cfg['mpc']['prediction_horizon'],
            control_horizon=cfg['mpc']['control_horizon']
        )
        
        self.setpoint = cfg['mpc']['setpoint']
        self.action_limit = cfg['rl'].get('action_limit', 0.1)

        # RL aksiyonu: [-1, 1] arası
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Gözlem uzayı: [Hata, Değişim, Oksijen, Mevcut Yakıt]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.prev_temp = 1400.0
        self.last_action = 0.0
        self.step_count = 0
        self.max_steps = 5000

    def _get_obs(self):
        # Normalizasyon sinir ağının öğrenmesini kolaylaştırır
        return np.array([
            (self.plant.temp - self.setpoint) / 100.0,
            (self.plant.temp - self.prev_temp) / 10.0,
            self.plant.o2 / 10.0,
            self.plant.fuel / 20.0
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        u_mpc = self.mpc.optimize(self.plant)
        
        # RL Residual (Düzeltme)
        rl_residual = float(action[0]) * self.action_limit
        total_fuel = u_mpc + rl_residual

        self.prev_temp = self.plant.temp
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)

        # --- REWARD (Metric Optimizer Version) ---
        error = temp - self.setpoint
        
        # 1. Hassaslık (Scaled by 100 to lower Value Loss)
        reward = -(abs(error) / 100.0) 
        
        # 2. Yumuşaklık (Smoothness)
        action_diff = abs(float(action[0]) - self.last_action)
        reward -= action_diff * 0.01 

        self.last_action = float(action[0])

        # Terminasyon mantığı (Overfit engelleyici)
        terminated = False 
        if abs(error) > 500: # Güvenlik sınırı
            terminated = True
            reward -= 10.0 # Ceza ölçeklendirildi

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        self.prev_temp = 1400.0
        self.last_action = 0.0
        self.step_count = 0
        return self._get_obs(), {}

# =================================================================
# 2. WINDOWS WORKER DESTEĞİ
# =================================================================
def make_env(cfg):
    """Windows multiprocessing için lambda yerine açık fonksiyon"""
    def _init():
        return ResidualKilnEnv(cfg)
    return _init

# =================================================================
# 3. EĞİTİM VE ANA AKIŞ
# =================================================================
if __name__ == "__main__":
    # Windows için kritik destek
    multiprocessing.freeze_support()

    # Manuel Config (Dilersen load_config ile dosyadan çekebilirsin)
    config = {
        'hardware': {
            'num_cpu': 8, 
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'mpc': {
            'prediction_horizon': 20,
            'control_horizon': 5,
            'setpoint': 1450.0
        },
        'rl': {
            'total_timesteps': 100000,
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 256,
            'gamma': 0.99,
            'action_limit': 0.1
        },
        'paths': {
            'model_name': "models/ppo_kiln_hybrid_v3"
        }
    }

    # Klasörleri oluştur
    os.makedirs("models", exist_ok=True)

    # Vektörize edilmiş paralel ortamları başlat
    print(f"🚀 {config['hardware']['num_cpu']} çekirdek ile ortamlar hazırlanıyor...")
    env = SubprocVecEnv([make_env(config) for _ in range(config['hardware']['num_cpu'])])

    # PPO Modelini Tanımla
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config['rl']['learning_rate'],
        n_steps=config['rl']['n_steps'],
        batch_size=config['rl']['batch_size'],
        gamma=config['rl']['gamma'],
        ent_coef=0.01,
        device=config['hardware']['device']
    )

    print("🧠 Hibrit Zeka Eğitimi Başlıyor (100k step)...")
    try:
        model.learn(total_timesteps=config['rl']['total_timesteps'])
        # Kaydet
        model.save(config['paths']['model_name'])
        print(f"✅ Başarılı! Model kaydedildi: {config['paths']['model_name']}")
    except Exception as e:
        print(f"❌ Eğitim sırasında hata oluştu: {e}")
    finally:
        env.close()
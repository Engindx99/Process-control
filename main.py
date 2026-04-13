import os
import yaml
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Kendi modüllerin
from src.digital_twin.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

# =================================================================
# 1. ENVIRONMENT CLASS (Windows pickle sorunu için global scope'ta)
# =================================================================
class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super(ResidualKilnEnv, self).__init__()
        self.cfg = cfg
        
        # Plant ve MPC başlatma
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(
            prediction_horizon=cfg['mpc']['prediction_horizon'],
            control_horizon=cfg['mpc']['control_horizon']
        )
        
        self.setpoint = cfg['mpc']['setpoint']
        self.action_limit = cfg['rl'].get('action_limit', 0.1)

        # Space tanımlamaları (Hatanın çözümü için burada olması şart)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.prev_temp = 1400.0
        self.last_action = 0.0
        self.step_count = 0
        self.max_steps = 2000

    def _get_obs(self):
        return np.array([
            (self.plant.temp - self.setpoint) / 100.0,
            (self.plant.temp - self.prev_temp) / 10.0,
            self.plant.o2 / 10.0,
            self.plant.fuel / 20.0
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        u_mpc = self.mpc.optimize(self.plant)
        
        # RL katkısı
        rl_residual = float(action[0]) * self.action_limit
        total_fuel = u_mpc + rl_residual

        self.prev_temp = self.plant.temp
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)

        # Reward Mekanizması
        error = temp - self.setpoint
        reward = -(abs(error) * 0.5)
        
        action_diff = abs(float(action[0]) - self.last_action)
        reward -= action_diff * 0.1 

        self.last_action = float(action[0])

        # Overfit engelleyici: terminated daima False (max_steps'e kadar devam)
        terminated = False
        if abs(error) > 500: # Güvenlik sınırı
            terminated = True
            reward -= 100.0

        truncated = self.step_count >= self.max_steps
        
        # MPC loglama (Eğer MPC sınıfında log_step varsa)
        if hasattr(self.mpc, 'log_step'):
            self.mpc.log_step(self.step_count, total_fuel, temp, o2)

        return self._get_obs(), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        self.prev_temp = 1400.0
        self.last_action = 0.0
        self.step_count = 0
        return self._get_obs(), {}

# =================================================================
# 2. YARDIMCI FONKSİYONLAR
# =================================================================
def make_env(cfg):
    """Windows multiprocessing için lambda yerine fonksiyon kullanımı"""
    def _init():
        return ResidualKilnEnv(cfg)
    return _init

def train_hybrid_model(cfg):
    # Paralel ortamları başlat
    env = SubprocVecEnv([make_env(cfg) for _ in range(cfg['hardware']['num_cpu'])])
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=cfg['rl']['learning_rate'],
        n_steps=cfg['rl']['n_steps'],
        batch_size=cfg['rl']['batch_size'],
        gamma=cfg['rl']['gamma'],
        ent_coef=0.01,
        device=cfg['hardware']['device']
    )
    
    model.learn(total_timesteps=cfg['rl']['total_timesteps'])
    return model

def run_hybrid_test(model, cfg):
    """Test için tekil ortam simülasyonu"""
    env = ResidualKilnEnv(cfg)
    obs, _ = env.reset()
    history = []
    
    for i in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        # Mevcut durumu kaydetmek için step öncesi u_mpc'yi alalım
        u_mpc = env.mpc.optimize(env.plant)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        history.append({
            "step": i,
            "temp": float(env.plant.temp),
            "fuel": float(env.plant.fuel),
            "u_mpc": float(u_mpc),
            "u_rl": float(action[0] * cfg['rl']['action_limit']),
            "error": float(env.plant.temp - cfg['mpc']['setpoint'])
        })
        if terminated or truncated: break
            
    return history, env

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# =================================================================
# 3. ANA AKIŞ
# =================================================================
if __name__ == "__main__":
    # Windows için multiprocessing desteği (Kritik!)
    import multiprocessing
    multiprocessing.freeze_support()

    cfg = load_config()
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    model_path = cfg['paths']['model_name']
    
    # Eğitim veya Yükleme
    if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
        print("✅ Mevcut model yükleniyor...")
        model = PPO.load(model_path, device=cfg['hardware']['device'])
    else:
        print("🚀 Yeni hibrit eğitim başlıyor (100k step)...")
        model = train_hybrid_model(cfg)
        model.save(model_path)
        print("💾 Model kaydedildi.")

    # Test ve Kayıt
    print("📊 Performans test ediliyor...")
    history, env = run_hybrid_test(model, cfg)

    df = pd.DataFrame(history)
    df.to_csv(cfg['paths']['results_csv'], index=False)
    
    if hasattr(env.mpc, 'save'):
        env.mpc.save(cfg['paths']['log_json'])

    print(f"🏁 İşlem tamamlandı. Sonuçlar: {cfg['paths']['results_csv']}")
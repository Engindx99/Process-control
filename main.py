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

# Kendi modüllerin (Dosya yollarının doğruluğundan emin ol)
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
        self.max_steps = 2000

    def _get_obs(self):
        # Girdileri normalize etmek sinir ağının (MLP) daha hızlı öğrenmesini sağlar
        return np.array([
            (self.plant.temp - self.setpoint) / 100.0,
            (self.plant.temp - self.prev_temp) / 10.0,
            self.plant.o2 / 10.0,
            self.plant.fuel / 20.0
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        # MPC ana aksiyonu
        u_mpc = self.mpc.optimize(self.plant)
        
        # RL Residual (Düzeltme)
        rl_residual = float(action[0]) * self.action_limit
        total_fuel = u_mpc + rl_residual

        self.prev_temp = self.plant.temp
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)

        # --- REWARD (Kritik Ölçeklendirme) ---
        error = temp - self.setpoint
        
        # Hatayı 100'e bölerek 'Value Loss'un patlamasını engelliyoruz
        # Amacımız explained_variance metriğini 10^-6'dan anlamlı seviyelere çekmek
        reward = -(abs(error) / 100.0) 
        
        # Aksiyon sarsıntı cezası (Daha pürüzsüz kontrol için)
        action_diff = abs(float(action[0]) - self.last_action)
        reward -= action_diff * 0.05 

        self.last_action = float(action[0])

        # Güvenlik ve Terminasyon
        terminated = False 
        if abs(error) > 500: # Fırın kontrolden çıkarsa
            terminated = True
            reward -= 5.0 # Yeni ödül skalasına uygun ceza

        truncated = self.step_count >= self.max_steps

        # Opsiyonel: MPC loglama
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
# 2. YARDIMCI FONKSİYONLAR & MULTIPROCESSING
# =================================================================
def make_env(cfg):
    """Windows worker'ları için serileştirilebilir ortam başlatıcı"""
    def _init():
        return ResidualKilnEnv(cfg)
    return _init

def load_config():
    """Config dosyasını yükle veya default dön"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Dosya yoksa manuel tanımlama
        return {
            'hardware': {'num_cpu': 8, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            'mpc': {'prediction_horizon': 20, 'control_horizon': 5, 'setpoint': 1450.0},
            'rl': {
                'total_timesteps': 100000, 
                'learning_rate': 0.0003, 
                'n_steps': 2048, 
                'batch_size': 256, 
                'gamma': 0.99, 
                'action_limit': 0.1
            },
            'paths': {
                'model_name': "models/ppo_kiln_hybrid_v3",
                'results_csv': "data/results.csv",
                'log_json': "data/log.json"
            }
        }

# =================================================================
# 3. ANA AKIŞ
# =================================================================
if __name__ == "__main__":
    # Windows'ta paralel eğitim için BU SATIR ŞARTTIR
    multiprocessing.freeze_support()

    cfg = load_config()
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    model_path = cfg['paths']['model_name']

    # Paralel Ortam Hazırlığı
    print(f"🚀 {cfg['hardware']['num_cpu']} çekirdek ile paralel ortamlar başlatılıyor...")
    env = SubprocVecEnv([make_env(cfg) for _ in range(cfg['hardware']['num_cpu'])])

    # Model Eğitimi veya Yükleme
    if os.path.exists(model_path + ".zip"):
        print("✅ Mevcut model bulundu, yükleniyor...")
        model = PPO.load(model_path, env=env, device=cfg['hardware']['device'])
    else:
        print("🧠 Yeni hibrit model eğitimi başlıyor...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=cfg['rl']['learning_rate'],
            n_steps=cfg['rl']['n_steps'],
            batch_size=cfg['rl']['batch_size'],
            gamma=cfg['rl']['gamma'],
            ent_coef=0.01, # Exploration'ı canlı tutar
            device=cfg['hardware']['device']
        )
        model.learn(total_timesteps=cfg['rl']['total_timesteps'])
        model.save(model_path)
        print(f"💾 Model kaydedildi: {model_path}")

    # --- TEST AŞAMASI ---
    print("📊 Eğitim sonrası test simülasyonu yapılıyor...")
    test_env = ResidualKilnEnv(cfg)
    obs, _ = test_env.reset()
    history = []

    for i in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        u_mpc_val = test_env.mpc.optimize(test_env.plant)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        
        history.append({
            "step": i,
            "temp": float(test_env.plant.temp),
            "fuel": float(test_env.plant.fuel),
            "u_mpc": float(u_mpc_val),
            "u_rl_residual": float(action[0] * cfg['rl']['action_limit']),
            "error": float(test_env.plant.temp - cfg['mpc']['setpoint'])
        })
        if terminated or truncated: break

    # Kayıt işlemleri
    pd.DataFrame(history).to_csv(cfg['paths']['results_csv'], index=False)
    print(f"🏁 Tamamlandı. Veriler kaydedildi: {cfg['paths']['results_csv']}")
    
    env.close()
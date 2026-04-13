import os
import yaml
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import sys

# --- 0. PATH AYARI ---
# Modüllerin (src) bulunabilmesi için ana dizini ekliyoruz
sys.path.append(os.getcwd())

from src.digital_twin.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

# --- 1. CONFIG YÜKLEME ---
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

# --- 2. RESIDUAL RL ORTAMI ---
class ResidualKilnEnv(gym.Env):
    def __init__(self):
        super(ResidualKilnEnv, self).__init__()
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(
            prediction_horizon=cfg['mpc']['prediction_horizon'], 
            control_horizon=cfg['mpc']['control_horizon']
        )
        self.setpoint = cfg['mpc']['setpoint']
        
        limit = cfg['rl']['action_limit']
        self.action_space = spaces.Box(low=-limit, high=limit, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.prev_temp = 1400.0
        self.last_action = np.array([0.0], dtype=np.float32)

    def step(self, action):
        u_mpc = self.mpc.optimize(self.plant)
        total_fuel = u_mpc + action[0]
        
        self.prev_temp = self.plant.temp
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)
        
        error = abs(temp - self.setpoint)
        
        # Reward Fonksiyonu
        reward = -(error**2 * 0.1) 
        action_diff = abs(action[0] - self.last_action[0])
        reward -= action_diff * 10.0 
        reward -= abs(action[0]) * 1.0
        
        if error < 1.0:
            reward += (1.0 - error) * 5.0 
            
        self.last_action = action
        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        return np.array([
            self.plant.temp - self.setpoint,
            self.plant.temp - self.prev_temp,
            self.plant.o2,
            self.plant.fuel
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        self.prev_temp = 1400.0
        self.last_action = np.array([0.0], dtype=np.float32)
        return self._get_obs(), {}

def make_env():
    return ResidualKilnEnv()

# --- 3. ANA ÇALIŞTIRICI ---
if __name__ == "__main__":
    # Gerekli klasörleri oluştur
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("experiments/plots", exist_ok=True)

    model_path = cfg['paths']['model_name'] 
    zip_path = model_path if model_path.endswith(".zip") else model_path + ".zip"

    # --- MODEL YÜKLEME VEYA EĞİTİM ---
    if os.path.exists(zip_path):
        print(f"✅ Kayıtlı model bulundu: {zip_path}. Yükleniyor...")
        model = PPO.load(model_path, device=cfg['hardware']['device'])
    else:
        fallback = "ppo_kiln_residual_model.zip"
        if os.path.exists(fallback):
            print(f"⚠️ Model ana dizinde bulundu, {model_path} konumuna taşınıyor...")
            os.rename(fallback, zip_path)
            model = PPO.load(model_path, device=cfg['hardware']['device'])
        else:
            print(f"🚀 Model bulunamadı. Yeni eğitim başlıyor ({cfg['hardware']['num_cpu']} çekirdek)...")
            env = SubprocVecEnv([make_env for _ in range(cfg['hardware']['num_cpu'])])
            
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
            model.save(model_path)
            env.close() # Kaynakları serbest bırak
            print(f"💾 Eğitim tamamlandı: {zip_path}")

    # --- TEST VE VERİ ÜRETİMİ ---
    print("📊 Test simülasyonu başlatılıyor...")
    test_env = ResidualKilnEnv()
    obs, _ = test_env.reset()
    history = []

    # 5000 adım: Fırın dinamiğini uzun vadeli görmek için
    for i in range(5000):
        # Kıyaslama için saf MPC kararını al
        u_mpc_base = test_env.mpc.optimize(test_env.plant)
        
        # RL tahmini (Deterministic=True: En iyi aksiyonu seç)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = test_env.step(action)
        
        history.append({
            "step": i, 
            "temp": float(test_env.plant.temp), 
            "u_rl_residual": float(action[0]),
            "u_mpc_base": float(u_mpc_base),
            "total_fuel": float(test_env.plant.fuel),
            "error": float(test_env.plant.temp - test_env.setpoint)
        })

    # Sonuçları Kaydet
    df = pd.DataFrame(history)
    df.to_csv(cfg['paths']['results_csv'], index=False)
    test_env.mpc.save(cfg['paths']['log_json'])
    
    print(f" Başarılı! Veriler '{cfg['paths']['results_csv']}' dosyasına yazıldı.")
    print(" Şimdi 'python eval.py' komutuyla 3 panelli analiz grafiğini oluşturabilirsin.")
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Kendi modüllerinden importlar
from digital_twin.dt import RotaryKilnDigitalTwin
from mpc.mpc import MPC

# --- 1. CONFIG YÜKLEME ---
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

# --- 2. RL ORTAMI ---
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
        
        # Ödül Fonksiyonu
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

# --- 3. PARALEL ÇALIŞTIRMA İÇİN DÜZELTİLMİŞ FONKSİYON ---
def make_env():
    # Windows'ta SubprocVecEnv bazen doğrudan fonksiyon ister
    return ResidualKilnEnv()

# --- 4. ANA ÇALIŞTIRICI ---
if __name__ == "__main__":
    model_name_path = cfg['paths']['model_name']
    model_zip = model_name_path + ".zip"

    if os.path.exists(model_zip):
        print(f"✅ Model yüklendi: {model_zip}")
        model = PPO.load(model_name_path, device=cfg['hardware']['device'])
    else:
        print(f"🚀 Eğitim başlıyor ({cfg['hardware']['num_cpu']} çekirdek)...")
        # DÜZELTME: [make_env for _ in ...] yerine lambda veya liste içine çağrılmamış hali
        env = SubprocVecEnv([make_env for _ in range(cfg['hardware']['num_cpu'])])
        
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=float(cfg['rl']['learning_rate']), # YAML'dan bazen float/str karışabilir
            n_steps=cfg['rl']['n_steps'],
            batch_size=cfg['rl']['batch_size'],
            gamma=cfg['rl']['gamma'],
            device=cfg['hardware']['device']
        )
        
        model.learn(total_timesteps=cfg['rl']['total_timesteps'])
        model.save(model_name_path)

    # --- TEST VE KAYIT ---
    print("📊 Test ve Kayıt işlemi...")
    test_env = ResidualKilnEnv()
    obs, _ = test_env.reset()
    history = []

    # Test süresini 5000 adıma çıkardın, bu fırın kararlılığını görmek için çok iyi
    for i in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, _ = test_env.step(action)
        
        test_env.mpc.log_step(i, test_env.plant.fuel, test_env.plant.temp, test_env.plant.o2)
        history.append({
            "step": i, 
            "temp": test_env.plant.temp, 
            "u_rl": action[0],
            "fuel": test_env.plant.fuel
        })

    test_env.mpc.save(cfg['paths']['log_json'])
    pd.DataFrame(history).to_csv(cfg['paths']['results_csv'], index=False)
    
    # Grafik çıktılarını daha detaylı görelim
    plt.figure(figsize=(12,6))
    plt.plot([h['temp'] for h in history], label='Fırın Sıcaklığı', color='blue')
    plt.axhline(y=1450, color='r', linestyle='--', label='Set Point (1450°C)')
    plt.title("Eğitim Sonrası Hibrit (MPC+RL) Kontrol Performansı")
    plt.xlabel("Adım")
    plt.ylabel("Sıcaklık (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
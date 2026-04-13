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
        # Config'den gelen MPC ayarları
        self.mpc = MPC(
            prediction_horizon=cfg['mpc']['prediction_horizon'], 
            control_horizon=cfg['mpc']['control_horizon']
        )
        self.setpoint = cfg['mpc']['setpoint']
        
        limit = cfg['rl']['action_limit']
        self.action_space = spaces.Box(low=-limit, high=limit, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.prev_temp = 1400.0

    def step(self, action):
        u_mpc = self.mpc.optimize(self.plant)
        total_fuel = u_mpc + action[0]
        
        self.prev_temp = self.plant.temp
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)
        
        error = abs(temp - self.setpoint)
        # Ödül fonksiyonunu kararlılık odaklı güncelledik
        reward = -(error * 0.2) - (abs(action[0]) * 5.0)
        if error < 1.0: reward += 10.0 # Hedefe yakınsa büyük ödül
        
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
        return self._get_obs(), {}

def make_env():
    return ResidualKilnEnv()

# --- 3. ANA ÇALIŞTIRICI ---
if __name__ == "__main__":
    model_zip = cfg['paths']['model_name'] + ".zip"

    # MODEL KONTROLÜ
    if os.path.exists(model_zip):
        print(f"✅ Model yüklendi: {model_zip}")
        model = PPO.load(cfg['paths']['model_name'], device=cfg['hardware']['device'])
    else:
        print(f"🚀 Eğitim başlıyor ({cfg['hardware']['num_cpu']} çekirdek)...")
        env = SubprocVecEnv([make_env for _ in range(cfg['hardware']['num_cpu'])])
        
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=cfg['rl']['learning_rate'],
            n_steps=cfg['rl']['n_steps'],
            batch_size=cfg['rl']['batch_size'],
            gamma=cfg['rl']['gamma'],
            device=cfg['hardware']['device']
        )
        
        model.learn(total_timesteps=cfg['rl']['total_timesteps'])
        model.save(cfg['paths']['model_name'])

    # --- TEST VE KAYIT ---
    print("📊 Test ve Kayıt işlemi...")
    test_env = ResidualKilnEnv()
    obs, _ = test_env.reset()
    history = []

    for i in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, _ = test_env.step(action)
        
        test_env.mpc.log_step(i, test_env.plant.fuel, test_env.plant.temp, test_env.plant.o2)
        history.append({"step": i, "temp": test_env.plant.temp, "u_rl": action[0]})

    test_env.mpc.save(cfg['paths']['log_json'])
    pd.DataFrame(history).to_csv(cfg['paths']['results_csv'], index=False)
    
    # Grafik
    plt.plot([h['temp'] for h in history])
    plt.axhline(y=1450, color='r', linestyle='--')
    plt.show()
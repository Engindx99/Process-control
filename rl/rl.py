import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# Mevcut modüllerinden importlar
from digital_twin.dt import RotaryKilnDigitalTwin
from mpc.mpc import MPC

# --- 1. RL ORTAMI (WRAPPER) ---
class ResidualKilnEnv(gym.Env):
    def __init__(self):
        super(ResidualKilnEnv, self).__init__()
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC()
        self.setpoint = 1450.0
        
        # RL Aksiyonu: MPC sinyaline yapılacak ince ayar (±0.2 m3/h)
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(1,), dtype=np.float32)
        
        # Gözlem: [Hata, Sıcaklık Değişimi, Mevcut O2, MPC Yakıtı]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.prev_temp = 1400.0

    def _get_obs(self):
        return np.array([
            self.plant.temp - self.setpoint,
            self.plant.temp - self.prev_temp,
            self.plant.o2,
            self.plant.fuel
        ], dtype=np.float32)

    def step(self, action):
        # MPC Kararı
        u_mpc = self.mpc.optimize(self.plant)
        
        # Hibrit Karar (MPC + RL)
        total_fuel = u_mpc + action[0]
        
        # Simülasyon Adımı
        self.prev_temp = self.plant.temp
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)
        
        # Ödül: Hata karesi cezası + Verimlilik ödülü - Gereksiz RL müdahale cezası
        error = abs(temp - self.setpoint)
        reward = -(error**2 * 0.05) + (eff * 10) - (abs(action[0]) * 5)
        
        return self._get_obs(), reward, False, False, {}

    def reset(self, seed=None, options=None):
        if seed is not None: set_random_seed(seed)
        self.plant = RotaryKilnDigitalTwin()
        self.prev_temp = 1400.0
        return self._get_obs(), {}

# --- 2. PARALEL ORTAM YARDIMCISI ---
def make_env(rank, seed=0):
    def _init():
        env = ResidualKilnEnv()
        env.reset(seed=seed + rank)
        return env
    return _init

# --- 3. EĞİTİM VE TEST ---
def run_parallel_training(num_cpu=11):
    model_path = "ppo_kiln_hybrid"
    
    # Paralel ortamları oluştur
    print(f"\n> {num_cpu} çekirdek üzerinde paralel simülasyonlar başlatılıyor...")
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # Modeli oluştur (PPO)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048)
    
    print("> Eğitim başladı. İşlemci kullanımını şimdi kontrol edebilirsin.")
    model.learn(total_timesteps=150000) # 150k adım 14 çekirdekle hızlı biter
    model.save(model_path)
    print(f"> Model kaydedildi: {model_path}")
    
    return model_path

if __name__ == "__main__":
    # 1. EĞİTİM
    # Not: Multiprocessing için Windows'ta bu blok şarttır.
    m_path = run_parallel_training(num_cpu=12)
    
    # 2. GÖRSELLEŞTİRME İÇİN TEK BİR TEST KOŞUSU
    print("\n> Eğitim sonrası performans testi yapılıyor...")
    test_env = ResidualKilnEnv()
    model = PPO.load(m_path)
    
    history = []
    obs, _ = test_env.reset()
    for i in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        # Test sırasında bozucu etkiler ekleyelim
        if 1000 < i < 1100: test_env.plant.temp -= 4.0
        
        # Adım atarken MPC + RL birleşik çalışır (Env içinde)
        obs, reward, _, _, _ = test_env.step(action)
        
        history.append({
            "step": i,
            "temp": test_env.plant.temp,
            "u_rl": action[0],
            "fuel": test_env.plant.fuel
        })

    # Grafikleme
    df = pd.DataFrame(history)
    plt.figure(figsize=(12, 5))
    plt.plot(df['step'], df['temp'], color='red', label='Sıcaklık (MPC+RL)')
    plt.axhline(y=1450, color='black', linestyle='--')
    plt.title("Eğitim Sonrası Hibrit Kontrol Performansı")
    plt.legend()
    plt.show()
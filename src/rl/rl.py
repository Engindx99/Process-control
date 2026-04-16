import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import os
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Proje yapısına göre importlar
from src.dt.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(cfg)
        
        # 100 kütle için ideal etki gücü (0.18 - 0.25 arası)
        self.action_limit = cfg['rl'].get('action_limit', 0.18)
        
        # [Yakıt Düzeltme, Fan Düzeltme]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Gözlem (14 Boyutlu): [Hata, Trend, O2, Eff] + [Yakıt Geçmişi t-1...t-10]
        # Hafıza sayesinde RL "yoldaki" yakıtı görüp gecikmeyi sezebilir.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        self.reset()

    def _get_obs(self, eff=0.8):
        temp_error = (self.plant.temp - 1450.0)
        
        # Normalizasyon: Hatalar ±50 civarı olduğu için 50'ye bölmek veriyi [-1, 1] arasına çeker.
        base_obs = [
            temp_error / 50.0, 
            (self.plant.temp - self.prev_temp) / 2.0,
            (self.plant.o2 - 3.0) / 2.0,
            eff
        ]
        
        # Son 10 adımlık yakıt geçmişi (Hafıza katmanı)
        fuel_memory = np.array(self.plant.fuel_history[-10:], dtype=np.float32) / 5.0
        
        return np.concatenate([base_obs, fuel_memory]).astype(np.float32)

    def step(self, action):
        # 1. Kontrolcü Kararları (MPC Ana Kontrolü Sağlar)
        u_mpc_f, u_mpc_v = self.mpc.optimize(self.plant)
        
        # RL İnce Ayarı (Residual Control)
        res_f = action[0] * self.action_limit
        res_v = action[1] * (self.action_limit * 50.0) 
        
        self.prev_temp = self.plant.temp
        
        # 2. Fiziksel Adım (Digital Twin üzerinde uygulama)
        current_fuel = u_mpc_f + res_f
        current_fan = u_mpc_v + res_v
        temp, o2, eff = self.plant.step(current_fuel, current_fan)
        
        # 3. YENİ ÖDÜL TASARIMI (Loss Patlamasını Önleyen Ölçekleme)
        temp_err = abs(temp - 1450.0)
        
        # A. Sıcaklık Cezası (Karesel ceza, 10 dereceden sonra devreye girer)
        if temp_err > 10.0:
            r_temp = -(temp_err ** 2) * 0.005 # 50 derece hata -> -12.5 ceza
        else:
            r_temp = -temp_err * 0.05 # 2 derece hata -> -0.1 ceza
            
        # B. Aksiyon Cezası (Gereksiz titremeyi/osilasyonu engeller)
        r_smooth = -np.mean(np.square(action)) * 2.0
        
        # C. Enerji Cezası (Yakıt ekonomisi için)
        r_energy = -current_fuel * 0.02
        
        # D. Kademeli Bonus (RL'in moralini yüksek tutmak için)
        bonus = 0.0
        if temp_err < 10.0:
            bonus += 1.0 # İlk hedef 10 derecenin içine girmek
            if temp_err < 2.0:
                bonus += 5.0 / (temp_err + 0.1) # İnce ayar başarısı
        
        reward = r_temp + r_smooth + r_energy + bonus + (eff * 0.5)

        # 4. Sonlanma ve Kısıtlar
        terminated = (temp < 1200 or temp > 1700)
        if terminated: 
            reward -= 50.0 
        
        if self.step_count % 500 == 0:
            print(f"DEBUG | T: {temp:.1f} | Err: {temp-1450:.2f} | R: {reward:.2f} | Fuel: {current_fuel:.2f}")

        self.step_count += 1
        truncated = self.step_count >= 3000
        
        return self._get_obs(eff), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        if seed is not None:
            np.random.seed(seed)
            # Rastgele başlangıç sıcaklığı (1410-1490)
            self.plant.temp = np.random.uniform(1410, 1490)
        self.prev_temp = self.plant.temp
        self.step_count = 0
        return self._get_obs(), {}

# =================================================================
# EĞİTİM MOTORU (PPO)
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
    
    # Çok çekirdekli eğitim ortamı
    env = SubprocVecEnv([make_env(cfg, i) for i in range(num_cpu)])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=cfg['rl'].get('learning_rate', 1e-4),
        n_steps=cfg['rl'].get('n_steps', 4096),
        batch_size=cfg['rl'].get('batch_size', 512),
        gamma=cfg['rl'].get('gamma', 0.995),
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.02, # Keşfi canlı tut
        verbose=1,
        device=cfg['hardware'].get('device', 'auto')
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 40000 // num_cpu), 
        save_path='./models/checkpoints/',
        name_prefix='hybrid_kiln_v5_opt'
    )

    print(f"🚀 V5: Kararlı Isıl Kütle (100) ve 11 Çekirdekli Eğitim Başlıyor...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    save_path = cfg['paths'].get('model_save_path', "models/ppo_kiln_v5_final")
    model.save(save_path)
    print(f"✅ Eğitim bitti. Model kaydedildi: {save_path}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_rl(config)
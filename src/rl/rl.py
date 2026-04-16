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
        self.action_limit = cfg['rl'].get('action_limit', 0.18)
        
        # [Yakıt Düzeltme, Fan Düzeltme]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Gözlem (14 Boyutlu): [Hata, Trend, O2, Eff] + [Yakıt Geçmişi t-1...t-10]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        self.reset()

    def _get_obs(self, eff=0.8):
        temp_error = (self.plant.temp - 1450.0)
        base_obs = [
            temp_error / 50.0, 
            (self.plant.temp - self.prev_temp) / 2.0,
            (self.plant.o2 - 3.0) / 2.0,
            eff
        ]
        fuel_memory = np.array(self.plant.fuel_history[-10:], dtype=np.float32) / 5.0
        return np.concatenate([base_obs, fuel_memory]).astype(np.float32)

    def step(self, action):
        # 1. MPC Kararı
        u_mpc_f, u_mpc_v = self.mpc.optimize(self.plant)
        
        # 2. RL Residual Kararı
        res_f = action[0] * self.action_limit
        res_v = action[1] * (self.action_limit * 50.0) 
        
        self.prev_temp = self.plant.temp
        
        # 3. Fiziksel Simülasyon
        current_fuel = u_mpc_f + res_f
        current_fan = u_mpc_v + res_v
        temp, o2, eff = self.plant.step(current_fuel, current_fan)
        
        # 4. ÖDÜL HESAPLAMA (GÜNCEL)
        temp_err = abs(temp - 1450.0)
        
        # A. Sıcaklık Cezası
        if temp_err > 10.0:
            r_temp = -(temp_err ** 2) * 0.005
        else:
            r_temp = -temp_err * 0.05

        # B. Damping (Osilasyon Sönümleyici - Zikzakları Keser)
        delta_t = abs(temp - self.prev_temp)
        r_damping = -delta_t * 0.2 
            
        # C. Aksiyon Cezası (Yumuşak hareketler için)
        r_smooth = -np.mean(np.square(action)) * 2.0
        
        # D. Enerji Cezası
        r_energy = -current_fuel * 0.02
        
        # E. Hassas Bonus (1450'ye mıknatıs etkisi)
        bonus = 0.0
        if temp_err < 10.0:
            bonus += 1.0
            if temp_err < 2.0:
                bonus += 5.0 / (temp_err + 0.1)
        
        # TOPLAM ÖDÜL (Hatasız Birleştirme)
        reward = r_temp + r_damping + r_smooth + r_energy + bonus + (eff * 0.5)

        # 5. Sonlanma
        terminated = (temp < 1200 or temp > 1700)
        if terminated: reward -= 50.0 
        
        if self.step_count % 500 == 0:
            print(f"DEBUG | T: {temp:.1f} | Err: {temp-1450:.2f} | R: {reward:.2f} | Damping: {r_damping:.2f}")

        self.step_count += 1
        truncated = self.step_count >= 3000
        
        return self._get_obs(eff), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        if seed is not None:
            np.random.seed(seed)
            self.plant.temp = np.random.uniform(1410, 1490)
        self.prev_temp = self.plant.temp
        self.step_count = 0
        return self._get_obs(), {}

# --- EĞİTİM MOTORU ---

def make_env(cfg, rank, seed=0):
    def _init():
        env = ResidualKilnEnv(cfg)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_rl(cfg):
    num_cpu = cfg['hardware'].get('num_cpu', 11)
    
    env = SubprocVecEnv([make_env(cfg, i) for i in range(num_cpu)])
    env = VecMonitor(env)

    # DİKKAT: Fine-tuning için mevcut en iyi modelini yüklüyoruz
    model_path = "models/ppo_kiln_v5_pro_extended_3k"
    
    if os.path.exists(model_path + ".zip"):
        print(f"🔄 Mevcut model ({model_path}) yükleniyor, ince ayar başlıyor...")
        model = PPO.load(model_path, env=env)
        # Cilalama için hızı düşürüyoruz
        model.learning_rate = 1e-5 
    else:
        print("⚠️ Mevcut model bulunamadı, sıfırdan eğitim başlıyor!")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 40000 // num_cpu), 
        save_path='./models/checkpoints/',
        name_prefix='fine_tuned_kiln'
    )

    # 150k adım osilasyonu sönümlemek için idealdir
    model.learn(total_timesteps=150000, callback=checkpoint_callback)
    
    model.save("models/ppo_kiln_v5_fine_tuned")
    print("✅ İnce ayarlı model kaydedildi: models/ppo_kiln_v5_fine_tuned")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_rl(config)
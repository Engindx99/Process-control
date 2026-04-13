import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.digital_twin.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

# =================================================================
# 1. ORTAM TANIMI (ENVIRONMENT)
# =================================================================
class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Fiziksel model ve ana kontrolcü
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(
            prediction_horizon=cfg['mpc']['prediction_horizon'],
            control_horizon=cfg['mpc']['control_horizon']
        )
        
        self.setpoint = cfg['mpc']['setpoint']
        self.action_limit = cfg['rl'].get('action_limit', 0.1) # %10 değişim payı daha sağlıklıdır

        # RL aksiyonu: [-1, 1] arası bir değer üretir
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Gözlem uzayı: [Hata, Değişim, Oksijen, Mevcut Yakıt]
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
        
        # MPC'den gelen ana aksiyon
        u_mpc = self.mpc.optimize(self.plant)

        # RL'den gelen düzeltme (Residual)
        # Aksiyon limitini config'den alıyoruz (Örn: 0.1)
        rl_residual = action[0] * self.action_limit

        # Toplam yakıt
        total_fuel = u_mpc + rl_residual

        # Bitkiyi simüle et
        self.prev_temp = self.plant.temp
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)

        # --- REWARD (Overfit Önleyici Yeni Versiyon) ---
        error = temp - self.setpoint
        
        # 1. Hassaslık ödülü (Hata arttıkça ceza artar)
        reward = -(abs(error) * 0.5) 
        
        # 2. Yumuşaklık cezası (Sarsıntılı yakıt kullanımını engellemek için)
        action_diff = abs(action[0] - self.last_action)
        reward -= action_diff * 0.1 

        self.last_action = float(action[0])

        # --- TERMINATION ---
        # Erken bitirmeyi sildik. Model gürültüde kalmayı öğrenmeli.
        terminated = False 
        
        # Güvenlik kapaması (Fırın kontrolden çıkarsa)
        if abs(error) > 500:
            terminated = True
            reward -= 100.0

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        self.prev_temp = 1400.0
        self.last_action = 0.0
        self.step_count = 0
        return self._get_obs(), {}

# =================================================================
# 2. EĞİTİM VE ÇALIŞTIRMA (MAIN)
# =================================================================
if __name__ == "__main__":
    # Senin config yapın
    config = {
        'hardware': {'num_cpu': 8, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        'mpc': {
            'prediction_horizon': 20,
            'control_horizon': 5, # 2'den 5'e çıkardık, daha esnek olur
            'setpoint': 1450.0
        },
        'rl': {
            'total_timesteps': 100000,
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 256,
            'gamma': 0.99,
            'action_limit': 0.1 # Daha geniş etki alanı
        },
        'paths': {
            'model_name': "models/ppo_kiln_hybrid_v2"
        }
    }

    # Çok çekirdekli eğitim başlat
    env = SubprocVecEnv([lambda: ResidualKilnEnv(config) for _ in range(config['hardware']['num_cpu'])])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config['rl']['learning_rate'],
        n_steps=config['rl']['n_steps'],
        batch_size=config['rl']['batch_size'],
        gamma=config['rl']['gamma'],
        ent_coef=0.01, # Modelin 'statik' kalmasını engeller
        device=config['hardware']['device']
    )

    print("Hibrit Model Eğitimi Başlıyor...")
    model.learn(total_timesteps=config['rl']['total_timesteps'])
    
    # Kaydet
    model.save(config['paths']['model_name'])
    print(f"Model kaydedildi: {config['paths']['model_name']}")
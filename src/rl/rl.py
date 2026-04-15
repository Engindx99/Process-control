import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Senin modüllerin
from src.dt.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(cfg)
        
        self.action_limit = cfg['rl'].get('action_limit', 0.18)
        
        # Action: [-1, 1] arası (Yakıt ve Fan düzeltmesi)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Obs: [Sıcaklık Hatası, Temp_Delta, O2 Hatası, Yakıt, Fan, Verimlilik]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def _get_obs(self, eff=0.8):
        # GÖZLEM GÜNCELLEMESİ: Hassasiyeti artırmak için paydaları küçülttük
        temp_error = (self.plant.temp - 1450.0) 
        o2_error = (self.plant.o2 - 3.0)
        
        return np.array([
            temp_error / 10.0,       # 100 yerine 10 (3 derecelik hata artık 0.3 sinyali verir)
            (self.plant.temp - self.prev_temp) / 5.0, 
            o2_error / 1.0,          # O2 hatasını daha belirgin ilet
            (self.plant.fuel - 17.5) / 5.0, 
            (self.plant.fan - 1050.0) / 250.0, 
            eff
        ], dtype=np.float32)

    def step(self, action):
        # 1. MPC Kararı
        u_mpc_f, u_mpc_v = self.mpc.optimize(self.plant)
        
        # 2. RL Residual Katkısı
        res_f = action[0] * self.action_limit
        res_v = action[1] * (self.action_limit * 150)
        
        self.prev_temp = self.plant.temp
        
        # 3. Fiziksel Adım
        temp, o2, eff = self.plant.step(u_mpc_f + res_f, u_mpc_v + res_v)
        
        # 4. REWARD GÜNCELLEMESİ (Sparse değil, daha keskin bir ödül)
        temp_err = abs(temp - 1450.0)
        o2_err = abs(o2 - 3.0)
        
        # Sıcaklık cezası (Logaritmik veya daha sert doğrusal)
        # MPC'nin takıldığı 2.8 derece artık hissedilir bir ceza olacak
        r_temp = -(temp_err / 2.0) 
        r_o2 = -(o2_err**2) * 1.5
        r_eff = eff * 5.0
        
        # Bonus: Eğer hata 1 derecenin altındaysa büyük ödül (Stabiliteyi ödüllendir)
        bonus = 2.0 if temp_err < 1.0 else 0.0
        
        reward = r_temp + r_o2 + r_eff + bonus
        
        self.step_count += 1
        
        # 5. Sonlanma Koşulları
        terminated = (abs(temp - 1450) > 200 or o2 < 1.0 or o2 > 9.0)
        truncated = self.step_count >= 2000 # 3000 çok uzun, 2000 veri çeşitliliği için daha iyi
        
        return self._get_obs(eff), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        # Reset anında küçük bir rastgelelik eklemek öğrenmeyi hızlandırır (Exploration)
        if seed is not None:
            self.plant.temp += np.random.uniform(-10, 10)
        
        self.prev_temp = self.plant.temp
        self.step_count = 0
        return self._get_obs(), {}

# Eğitim fonksiyonu aynı kalabilir ancak learning_rate ve n_steps uyarısı:
# n_steps * num_cpu = total batch size. 2048 * 11 = 22,528 adımda bir güncelleme yapar.
# Bu büyük bir sayı, öğrenme yavaş görünebilir ama daha stabildir.

# =================================================================
# 2. PARALEL EĞİTİM YARDIMCI FONKSİYONLARI
# =================================================================

def make_env(cfg, rank, seed=0):
    """
    SubprocVecEnv için ortam yaratan yardımcı fonksiyon
    """
    def _init():
        env = ResidualKilnEnv(cfg)
        # Her çekirdek için farklı seed veriyoruz
        env.reset(seed=seed + rank)
        return env
    return _init

def train_rl(cfg):
    num_cpu = cfg['hardware'].get('num_cpu', 11)
    total_timesteps = cfg['rl'].get('total_timesteps', 100000)
    device = cfg['hardware'].get('device', 'auto')

    print(f"🚀 {num_cpu} çekirdek üzerinde paralel eğitim başlıyor... Cihaz: {device}")

    # 1. Vektörize Ortamı Oluştur
    env = SubprocVecEnv([make_env(cfg, i) for i in range(num_cpu)])
    env = VecMonitor(env)

    # 2. PPO Modelini Yapılandır
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg['rl'].get('learning_rate', 0.0003),
        n_steps=cfg['rl'].get('n_steps', 2048),
        batch_size=cfg['rl'].get('batch_size', 256),
        gamma=cfg['rl'].get('gamma', 0.99),
        verbose=1,
        device=device,
        tensorboard_log="./logs/kiln_ppo_tensorboard/"
    )

    # 3. Kayıt (Callback) Ayarı
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_cpu, 1), 
        save_path='./models/checkpoints/',
        name_prefix='rl_model'
    )

    # 4. EĞİTİMİ BAŞLAT
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 5. Final Modeli Kaydet
    save_path = cfg['paths'].get('model_save_path', "models/ppo_kiln_residual_v2")
    model.save(save_path)
    print(f"✅ Eğitim tamamlandı. Model: {save_path}")
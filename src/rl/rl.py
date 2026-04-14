import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Proje ana dizinini yola ekleyelim (Import hatalarını önlemek için)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.digital_twin.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super(ResidualKilnEnv, self).__init__()
        self.cfg = cfg
        from src.digital_twin.dt import RotaryKilnDigitalTwin
        from src.mpc.mpc import MPC
        
        self.plant = RotaryKilnDigitalTwin()
        self.mpc = MPC(cfg) 
        
        self.setpoint = cfg['mpc']['setpoint']
        self.action_limit = cfg['rl'].get('action_limit', 0.1)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Gözlem uzayını 5'e çıkaralım: [Hata, Delta_Temp, O2, Yakıt, Verimlilik]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.prev_temp = cfg['simulation'].get('initial_temp', 1400.0)
        self.last_action = 0.0
        self.step_count = 0
        self.max_steps = cfg['simulation'].get('total_steps', 3000)

    def _get_obs(self, eff=0.5):
        # Sayıları küçültmek (Normalization) PPO için hayati önem taşır
        return np.array([
            (self.plant.temp - self.setpoint) / 200.0, # Hata ölçeği
            (self.plant.temp - self.prev_temp) / 20.0, # Değişim hızı
            (self.plant.o2 - 3.0) / 5.0,               # O2'nin merkezden sapması
            (self.plant.fuel - 17.0) / 5.0,            # Yakıtın merkezden sapması
            eff                                        # Mevcut verimlilik (0-1 arası)
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        u_mpc = self.mpc.optimize(self.plant)
        rl_residual = float(action[0]) * self.action_limit
        total_fuel = np.clip(u_mpc + rl_residual, 12.0, 22.0)

        self.prev_temp = self.plant.temp
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)

        # --- YENİ ÖDÜL SİSTEMİ (Daha dar ölçek: -1 ile 1 arası hedefi) ---
        error = abs(temp - self.setpoint)
        
        # Sıcaklık ödülü (Sıfıra ne kadar yakınsa o kadar iyi)
        reward = -(error / 100.0) 
        
        # O2 ödülü (Sadece çok büyük sapmalarda ceza verelim ki kafası karışmasın)
        if abs(o2 - 3.0) > 1.0:
            reward -= 0.05
        
        # Verimlilik ödülü (Bunu bir bonus olarak ekleyelim)
        reward += (eff * 0.1)

        # Aksiyon yumuşatma
        action_diff = abs(float(action[0]) - self.last_action)
        reward -= action_diff * 0.01
        
        self.last_action = float(action[0])

        # Durdurma
        terminated = False 
        if error > 400:
            terminated = True
            reward -= 1.0 # Cezayı küçülttük (Gradyan patlamasını önlemek için)

        truncated = self.step_count >= self.max_steps
        
        return self._get_obs(eff), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = RotaryKilnDigitalTwin()
        self.prev_temp = self.cfg['simulation'].get('initial_temp', 1400.0)
        self.last_action = 0.0
        self.step_count = 0
        return self._get_obs(), {}

def make_env(cfg):
    def _init():
        return ResidualKilnEnv(cfg)
    return _init


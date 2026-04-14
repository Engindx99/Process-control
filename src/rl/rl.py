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

class ResidualKilnEnv(gym.Env):
    def __init__(self, cfg):
        super(ResidualKilnEnv, self).__init__()
        self.cfg = cfg
        
        # Fiziksel model
        self.plant = RotaryKilnDigitalTwin()
        
        # --- KRİTİK DÜZELTME: MPC artık tüm cfg'yi alıyor ---
        self.mpc = MPC(cfg) 
        
        self.setpoint = cfg['mpc']['setpoint']
        self.action_limit = cfg['rl'].get('action_limit', 0.1)

        # RL aksiyonu: [-1, 1] arası
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Gözlem uzayı
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.prev_temp = cfg['simulation'].get('initial_temp', 1400.0)
        self.last_action = 0.0
        self.step_count = 0
        self.max_steps = cfg['simulation'].get('total_steps', 5000)

    def _get_obs(self):
        return np.array([
            (self.plant.temp - self.setpoint) / 100.0,
            (self.plant.temp - self.prev_temp) / 10.0,
            self.plant.o2 / 10.0,
            self.plant.fuel / 20.0
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        # MPC ana yakıt değerini hesaplar
        u_mpc = self.mpc.optimize(self.plant)
        
        # RL Residual (Düzeltme) eklenir
        rl_residual = float(action[0]) * self.action_limit
        total_fuel = np.clip(u_mpc + rl_residual, 12.0, 22.0) # Güvenli sınır kısıtı

        self.prev_temp = self.plant.temp
        temp, o2, eff = self.plant.step(total_fuel, 1000.0)

        # --- REWARD ---
        error = temp - self.setpoint
        reward = -(abs(error) / 100.0) 
        
        # Yumuşaklık cezası
        action_diff = abs(float(action[0]) - self.last_action)
        reward -= action_diff * 0.01 
        
        # Oksijen güvenliği cezası (Ekstra güvenlik)
        if o2 < 2.5: reward -= 0.5

        self.last_action = float(action[0])

        terminated = False 
        if abs(error) > 500: # Kontrolden çıkma durumu
            terminated = True
            reward -= 5.0 

        truncated = self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, {}

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

# Alt kısımdaki if __name__ == "__main__" bloğunu 
# karmaşıklığı önlemek için silebilirsin veya boş bırakabilirsin.
import gymnasium as gym
import numpy as np
from mcp import RotaryKilnMPC # Yazdığın MPC sınıfını içe aktarıyoruz
from stable_baselines3 import PPO

class RotaryKilnEnv(gym.Env):
    def __init__(self):
        super(RotaryKilnEnv, self).__init__()
        
        # MPC ve Fiziksel Yapı
        self.controller = RotaryKilnMPC()
        
        # RL Aksiyon Alanı: MPC için Setpoint değişimi (-20 ile +20 derece arası müdahale)
        self.action_space = gym.spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32)
        
        # RL Gözlem Alanı: [Mevcut Sıcaklık, Önceki Yakıt Miktarı, Besleme Hızı]
        self.observation_space = gym.spaces.Box(
            low=np.array([500, 0, 0]), 
            high=np.array([1600, 15, 10]), 
            dtype=np.float32
        )
        
        self.state = np.array([1200.0, 5.0, 5.0], dtype=np.float32)
        self.target_base = 1400.0 # İdeal baz sıcaklık

    def step(self, action):
        current_temp = self.state[0]
        feed_rate = self.state[2]
        
        # 1. RL'den gelen aksiyonla yeni hedef sıcaklığı belirle
        target_temp = self.target_base + action[0]
        
        # 2. MPC'yi çalıştır ve optimal yakıt miktarını al
        # MPC burada kısıtlamaları (constraints) kontrol eder
        fuel_suggestion = self.controller.step(current_temp, target_temp, feed_rate)
        
        # 3. Fiziksel Simülasyonu ilerlet (Basitleştirilmiş bir adım)
        # Gerçekte bu kısım digital_twin.py'den beslenir
        new_temp = current_temp + (fuel_suggestion * 5) - (feed_rate * 2) + np.random.normal(0, 1)
        
        # 4. Ödül Hesaplama (Reward)
        # Ceza 1: Hedef sıcaklıktan sapma
        temp_error = abs(new_temp - target_temp)
        # Ceza 2: Yüksek yakıt tüketimi
        fuel_cost = fuel_suggestion * 0.5
        
        reward = -(temp_error * 0.1) - fuel_cost
        
        # State güncelleme
        self.state = np.array([new_temp, fuel_suggestion, feed_rate], dtype=np.float32)
        
        done = False # Endüstriyel süreçler genelde süreklidir
        truncated = False
        
        return self.state, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([1200.0, 5.0, 5.0], dtype=np.float32)
        return self.state, {}

# --- EĞİTİM BÖLÜMÜ ---
if __name__ == "__main__":
    env = RotaryKilnEnv()
    
    # PPO Algoritması ile eğitim
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    
    print("Eğitim başlıyor... MPC + RL hibrit model devrede.")
    model.learn(total_timesteps=10000)
    
    # Modeli kaydet
    model.save("../../models/kiln_rl_mpc_model")
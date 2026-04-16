import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.rl.rl import make_env
import yaml
import numpy as np

# 1. Konfigürasyon
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

env = make_env(cfg, seed=42, rank=0)()
model = PPO.load("models/ppo_kiln_v5_pro_extended_3k")

obs, _ = env.reset()
history = []

print("🔥 Veri doğrudan 'self.plant.temp' üzerinden çekiliyor...")

for i in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # GÖNDERDİĞİN KODDAKİ DEĞİŞKEN: self.plant.temp
    # env.unwrapped üzerinden plant nesnesine, oradan da temp'e ulaşıyoruz
    current_temp = env.unwrapped.plant.temp
    
    history.append(current_temp)

# 2. Görselleştirme
plt.figure(figsize=(15, 8))
plt.axhline(1450, color='red', linestyle='-', linewidth=2, label='HEDEF (1450°C)')

# Saf MPC (Varsa)
try:
    mpc = pd.read_csv("data/pure_mpc_results.csv")
    plt.plot(mpc['temp'][:3000].values, label='Saf MPC', color='gray', alpha=0.3, linestyle='--')
except:
    pass

# Hibrit (Mavi)
plt.plot(history, label='Yeni Hibrit (3k Extended)', color='blue', linewidth=1.5)

# Metrikler
mae = np.mean(np.abs(np.array(history) - 1450))
plt.title(f"Döner Fırın Kontrolü - Hibrit Performans\nMAE (Ortalama Hata): {mae:.2f}°C")
plt.xlabel("Zaman Adımı")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.grid(True, alpha=0.3)

# Grafiği 1450'ye odakla (Hata payına göre otomatik esner)
plt.ylim(1380, 1520) 
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.rl.rl import make_env
import yaml
import numpy as np

# 1. Konfigürasyon ve Model Yükleme
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

env = make_env(cfg, seed=42, rank=0)()
model = PPO.load("models/ppo_kiln_v5_final")

obs, _ = env.reset()
history = {
    'temp': [],
    'fuel': [],
    'fan': [],
    'action_fuel': [],
    'action_fan': []
}

print("🚀 Simülasyon başladı, 3000 adım koşturuluyor...")

for i in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Plant üzerinden verileri çekiyoruz
    plant = env.unwrapped.plant
    history['temp'].append(plant.temp)
    history['fuel'].append(plant.fuel)
    history['fan'].append(plant.fan)
    
    # Modelin verdiği ham aksiyonlar (0.18 limitli bölge)
    history['action_fuel'].append(action[0])
    history['action_fan'].append(action[1])

# --- GÖRSELLEŞTİRME ---

# PENCERE 1: Sıcaklık ve Yakıt Akışı (Üst Üste)
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Üst Grafik: Sıcaklık
ax1.axhline(1450, color='red', linestyle='--', linewidth=2, label='HEDEF (1450°C)')
ax1.plot(history['temp'], label='Burning Zone Sıcaklığı', color='blue', linewidth=1.5)
ax1.set_ylabel("Sıcaklık (°C)")
ax1.set_ylim(1400, 1500) # 1450'ye odaklı
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
mae = np.mean(np.abs(np.array(history['temp']) - 1450))
ax1.set_title(f"Sıcaklık Kontrolü Analizi (MAE: {mae:.2f}°C)")

# Alt Grafik: Yakıt Tüketimi
ax2.plot(history['fuel'], label='Anlık Yakıt Miktarı', color='darkorange', linewidth=1.5)
ax2.set_ylabel("Yakıt Akışı")
ax2.set_xlabel("Zaman Adımı")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.set_title("Yakıt Modülasyonu")

# PENCERE 2: RL Ajanı Aksiyon Analizi (Vana Hareketleri)
plt.figure(figsize=(15, 6))
plt.plot(history['action_fuel'], label='Yakıt Aksiyonu (Agent)', color='green', alpha=0.7)
plt.plot(history['action_fan'], label='Fan Aksiyonu (Agent)', color='purple', alpha=0.7)
plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
plt.title(f"RL Ajanı Vana Müdahaleleri (Action Limit: {cfg['rl'].get('action_limit', 0.18)})")
plt.xlabel("Zaman Adımı")
plt.ylabel("Aksiyon Şiddeti")
plt.legend()
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
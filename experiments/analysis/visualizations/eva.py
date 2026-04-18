import pandas as pd
import matplotlib.pyplot as plt
import yaml
import numpy as np
import os
from stable_baselines3 import PPO
from src.rl.rl import ResidualKilnEnv 

# =========================================================
# 1. CONFIG VE MODEL YÜKLEME
# =========================================================
# encoding='utf-8' ekleyerek Windows'un charmap hatasını çözüyoruz
with open("config.yaml", "r", encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

setpoint = cfg['system']['setpoint']
model_path = cfg['paths']['model_save_path']

# MPC Baseline Verisi
mpc_path = "data/pure_mpc_results.csv"
df_mpc = pd.read_csv(mpc_path) if os.path.exists(mpc_path) else None

# =========================================================
# 2. CANLI SİMÜLASYON (Yeni rl.py kurallarını çalıştırır)
# =========================================================
print("Güncel kurallarla canlı simülasyon başlatılıyor...")
env = ResidualKilnEnv(cfg)
model = PPO.load(model_path)

obs, _ = env.reset()
temp_rl, fuel_rl = [], []
n_steps = len(df_mpc) if df_mpc is not None else 1000

for _ in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    # Yeni Kalman filtreleme ve ceza mantığı bu step'in içinde çalışıyor
    obs, reward, terminated, truncated, _ = env.step(action)
    
    temp_rl.append(env.plant.temp)
    fuel_rl.append(env.plant.fuel)
    
    if terminated or truncated:
        break

temp_rl = np.array(temp_rl)
fuel_rl = np.array(fuel_rl)

# =========================================================
# 3. GÖRSELLEŞTİRME
# =========================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# --- SICAKLIK ---
ax1.axhline(setpoint, color='red', linestyle='--', label='Target')
if df_mpc is not None:
    mae_mpc = np.mean(np.abs(df_mpc['temp'] - setpoint))
    ax1.plot(df_mpc['temp'], label=f'Pure MPC (MAE: {mae_mpc:.2f})', color='gray', alpha=0.4)

mae_rl = np.mean(np.abs(temp_rl - setpoint))
ax1.plot(temp_rl, label=f'New Hybrid RL (MAE: {mae_rl:.2f})', color='#1f77b4', linewidth=1.25)
ax1.set_title("Fırın Sıcaklık: v8 Endüstriyel Mantık Analizi")
ax1.legend()

# --- YAKIT ---
if df_mpc is not None:
    ax2.plot(df_mpc['fuel'], label='MPC Fuel', color='gray', alpha=0.4, linestyle='--')

ax2.plot(fuel_rl, label='New Hybrid Fuel (Smoothed)', color='#2ca02c', linewidth=1.25)
ax2.set_title("Yakıt Akışı: Titreşim ve Stabilite Kontrolü")
ax2.set_xlabel("Steps")
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Final Test MAE: {mae_rl:.3f}")
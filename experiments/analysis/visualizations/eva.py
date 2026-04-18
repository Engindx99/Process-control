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
# encoding='utf-8' ile Windows karakter hatalarını önlüyoruz
with open("config.yaml", "r", encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

setpoint = cfg['system']['setpoint']
model_path = cfg['paths']['model_save_path']

# Zaman Kabulu: 1 Step = 5 Saniye
DT_SEC = 5 

# MPC Baseline Verisi (Karşılaştırma için)
mpc_path = "data/pure_mpc_results.csv"
df_mpc = pd.read_csv(mpc_path) if os.path.exists(mpc_path) else None

# =========================================================
# 2. CANLI SİMÜLASYON
# =========================================================
print("Güncel kurallarla canlı simülasyon başlatılıyor...")
env = ResidualKilnEnv(cfg)
model = PPO.load(model_path)

obs, _ = env.reset()
temp_rl, fuel_rl = [], []
n_steps = len(df_mpc) if df_mpc is not None else 3000

for _ in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    # Fiziksel model (Digital Twin) ve RL ödül mekanizması burada çalışır
    obs, reward, terminated, truncated, _ = env.step(action)
    
    temp_rl.append(env.plant.temp)
    fuel_rl.append(env.plant.fuel)
    
    if terminated or truncated:
        break

temp_rl = np.array(temp_rl)
fuel_rl = np.array(fuel_rl)

# Zaman dizisini oluştur (Adımları Dakikaya Çevir)
time_mins = np.arange(len(temp_rl)) * DT_SEC / 60

# =========================================================
# 3. GÖRSELLEŞTİRME (Zaman Odaklı ve Endüstriyel Renkler)
# =========================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# --- 3.1 SICAKLIK ANALİZİ ---
ax1.axhline(setpoint, color='red', linestyle='--', alpha=0.7, label=f'Hedef ({setpoint}°C)')

# 1090. Step (90.8. Dakika) Hedef İşaretçisi
target_min = (1090 * DT_SEC) / 60
ax1.axvline(target_min, color='blue', linestyle=':', alpha=0.5, label='Hedef Zaman (90.8 dk)')

if df_mpc is not None:
    mae_mpc = np.mean(np.abs(df_mpc['temp'] - setpoint))
    # MPC verisini de dakikaya çevirerek çiziyoruz
    time_mpc = np.arange(len(df_mpc)) * DT_SEC / 60
    ax1.plot(time_mpc, df_mpc['temp'], label=f'Pure MPC (MAE: {mae_mpc:.2f})', color='gray', alpha=0.4)

mae_rl = np.mean(np.abs(temp_rl - setpoint))
ax1.plot(time_mins, temp_rl, label=f'Hybrid RL (MAE: {mae_rl:.2f})', color='#1f77b4', linewidth=1.5)
ax1.set_title(f"Fırın Sıcaklık Analizi (v14) | MAE: {mae_rl:.3f}")
ax1.set_ylabel("Sıcaklık (°C)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# --- 3.2 YAKIT AKIŞI (SARI RENK VE STABİLİTE) ---
if df_mpc is not None:
    ax2.plot(time_mpc, df_mpc['fuel'], label='MPC Yakıt', color='gray', alpha=0.3, linestyle='--')

# İstediğin o profesyonel "Amber" sarısı: #FFBF00
ax2.plot(time_mins, fuel_rl, label='RL Yakıt (Smoothed)', color='#FFBF00', linewidth=1.8)
ax2.set_title("Yakıt Akışı Stabilite ve Pürüzsüzlük Kontrolü")
ax2.set_xlabel("Zaman (Dakika)")
ax2.set_ylabel("Yakıt Miktarı")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print(f"--- Simülasyon Tamamlandı ---")
print(f"Hedef Zaman: 90.8 dk (1090 step)")
print(f"RL Ortalama Hata (MAE): {mae_rl:.4f}")
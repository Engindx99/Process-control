import pandas as pd
import matplotlib.pyplot as plt
import yaml
import numpy as np
import os
from stable_baselines3 import PPO
from src.rl.rl import ResidualKilnEnv 

# =========================================================
# 1. YARDIMCI FONKSİYONLAR (KPI HESAPLAMA)
# =========================================================
def compute_control_kpis(temp_array, time_array, setpoint, band=2):
    """Sistem performans metriklerini hesaplar."""
    error = temp_array - setpoint
    
    # Settling Time
    within_band = np.abs(error) <= band
    settling_index = None
    for i in range(len(within_band)):
        if all(within_band[i:]):
            settling_index = i
            break
    settling_time = time_array[settling_index] if settling_index is not None else np.nan
    
    # Overshoot
    overshoot = np.max(temp_array) - setpoint
    
    # Rise Time (%10 -> %90)
    base_val = temp_array[0]
    span = setpoint - base_val
    try:
        t1 = time_array[np.where(temp_array >= base_val + span * 0.1)[0][0]]
        t2 = time_array[np.where(temp_array >= base_val + span * 0.9)[0][0]]
        rise_time = t2 - t1
    except:
        rise_time = np.nan

    # Kararlı Durum Hatası (MAE)
    mae = np.mean(np.abs(error))
    
    return {
        "ST": settling_time,
        "OS": overshoot,
        "RT": rise_time,
        "MAE": mae
    }

# =========================================================
# 2. CONFIG VE MODEL YÜKLEME
# =========================================================
with open("config.yaml", "r", encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

setpoint = cfg['system']['setpoint']
model_path = cfg['paths']['model_save_path']
DT_SEC = 5 

mpc_path = "data/pure_mpc_results.csv"
df_mpc = pd.read_csv(mpc_path) if os.path.exists(mpc_path) else None

# =========================================================
# 3. CANLI SİMÜLASYON
# =========================================================
print("RL Simülasyonu başlatılıyor...")
env = ResidualKilnEnv(cfg)
model = PPO.load(model_path)

obs, _ = env.reset()
temp_rl, fuel_rl = [], []
n_steps = len(df_mpc) if df_mpc is not None else 3000

for _ in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    temp_rl.append(env.plant.temp)
    fuel_rl.append(env.plant.fuel)
    if terminated or truncated: break

temp_rl = np.array(temp_rl)
time_rl = np.arange(len(temp_rl)) * DT_SEC / 60

# KPI Hesaplamaları
kpi_rl = compute_control_kpis(temp_rl, time_rl, setpoint)
if df_mpc is not None:
    time_mpc = np.arange(len(df_mpc)) * DT_SEC / 60
    kpi_mpc = compute_control_kpis(df_mpc['temp'].values, time_mpc, setpoint)

# =========================================================
# 4. GÖRSELLEŞTİRME (PROFESYONEL ANALİZ)
# =========================================================
plt.style.use('seaborn-v0_8-muted')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- 4.1 SICAKLIK VE KARŞILAŞTIRMALI KPI ---
ax1.axhline(setpoint, color='red', linestyle='--', alpha=0.6, label='Setpoint')

# RL Çizimi
ax1.plot(time_rl, temp_rl, label='Hybrid RL', color='#1f77b4', lw=2)

# MPC Çizimi (Varsa)
if df_mpc is not None:
    ax1.plot(time_mpc, df_mpc['temp'], label='Pure MPC', color='gray', alpha=0.4, linestyle='--')

# KPI Text Box (Karşılaştırmalı)
kpi_text = f"--- RL METRİKLERİ ---\nST: {kpi_rl['ST']:.1f} dk\nOS: {kpi_rl['OS']:.2f} °C\nMAE: {kpi_rl['MAE']:.2f} °C"
if df_mpc is not None:
    kpi_text += f"\n\n--- MPC METRİKLERİ ---\nST: {kpi_mpc['ST']:.1f} dk\nOS: {kpi_mpc['OS']:.2f} °C\nMAE: {kpi_mpc['MAE']:.2f} °C"

ax1.text(1.02, 0.5, kpi_text, transform=ax1.transAxes, verticalalignment='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ccc'), fontsize=10)

ax1.set_title("Sıcaklık Kontrol Performansı: RL vs MPC", fontsize=14)
ax1.set_ylabel("Sıcaklık (°C)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- 4.2 YAKIT VE STABİLİTE ---
if df_mpc is not None:
    ax2.plot(time_mpc, df_mpc['fuel'], color='gray', alpha=0.2, label='MPC Yakıt')

ax2.plot(time_rl, fuel_rl, color='#FFBF00', lw=1.5, label='RL Yakıt (Smoothed)')
ax2.set_title("Aktüatör (Yakıt) Kullanımı ve Kararlılık", fontsize=12)
ax2.set_xlabel("Zaman (Dakika)")
ax2.set_ylabel("Yakıt Akışı")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.subplots_adjust(right=0.85) # KPI kutusu için sağdan yer aç
plt.show()

print(f"Analiz Tamamlandı. RL MAE: {kpi_rl['MAE']:.4f}")
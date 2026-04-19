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
    overshoot = max(0, overshoot) # Negatif overshoot olmaz
    
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
    
    return {"ST": settling_time, "OS": overshoot, "RT": rise_time, "MAE": mae}

# =========================================================
# 2. CONFIG VE VERİ HAZIRLIĞI
# =========================================================
with open("config.yaml", "r", encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

setpoint = cfg['system']['setpoint']
model_path = cfg['paths']['model_save_path']
DT_SEC = cfg['system'].get('step_duration_sec', 5)

mpc_path = "data/pure_mpc_results.csv"
df_mpc = None

if os.path.exists(mpc_path):
    df_temp = pd.read_csv(mpc_path)
    # --- KRİTİK DÜZELTME: Kolon isimlerini standartlaştır ---
    df_temp.columns = [c.lower() for c in df_temp.columns]
    
    # 'temp' veya 'temperature' kolonunu bul ve standartlaştır
    if 'temperature' in df_temp.columns:
        df_temp['temp'] = df_temp['temperature']
    
    # 'fuel' kolonunu kontrol et
    if 'fuel' not in df_temp.columns and 'yakit' in df_temp.columns:
        df_temp['fuel'] = df_temp['yakit']
        
    df_mpc = df_temp
else:
    print(f"UYARI: {mpc_path} bulunamadı. Sadece RL çizilecek.")

# =========================================================
# 3. CANLI SİMÜLASYON (RL TEST)
# =========================================================
print("RL Simülasyonu başlatılıyor...")
env = ResidualKilnEnv(cfg)
# SB3 .zip uzantısını otomatik ekler, o yüzden dosya isminden temizliyoruz
load_path = model_path.replace(".zip", "")
model = PPO.load(load_path)

obs, _ = env.reset()
temp_rl, fuel_rl = [], []
# MPC verisi kadar simüle et veya varsayılan 1000 adım
n_steps = len(df_mpc) if df_mpc is not None else 1000

for i in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    # dt.py içindeki değişken isimlendirmesine göre güvenli alım
    temp_rl.append(env.plant.temp)
    fuel_rl.append(env.plant.fuel)
    
    if i % 100 == 0:
        print(f"Adım {i}/{n_steps} tamamlandı...")
    if terminated or truncated: break

temp_rl = np.array(temp_rl)
time_rl = (np.arange(len(temp_rl)) * DT_SEC) / 60

# KPI Hesaplamaları
kpi_rl = compute_control_kpis(temp_rl, time_rl, setpoint)
kpi_mpc = None

if df_mpc is not None:
    time_mpc = (np.arange(len(df_mpc)) * DT_SEC) / 60
    # Artık 'temp' kolonunun varlığından eminiz
    kpi_mpc = compute_control_kpis(df_mpc['temp'].values, time_mpc, setpoint)

# =========================================================
# 4. GÖRSELLEŞTİRME
# =========================================================
plt.style.use('seaborn-v0_8-muted')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- 4.1 SICAKLIK ANALİZİ ---
ax1.axhline(setpoint, color='red', linestyle='--', alpha=0.6, label=f'Setpoint ({setpoint}°C)')
ax1.plot(time_rl, temp_rl, label='Hybrid RL (MPC + Residual)', color='#1f77b4', lw=2)

if df_mpc is not None:
    ax1.plot(time_mpc, df_mpc['temp'], label='Pure MPC', color='gray', alpha=0.5, linestyle='--')

# KPI Text Box
kpi_text = f"--- RL METRİKLERİ ---\nMAE: {kpi_rl['MAE']:.3f} °C\nOS: {kpi_rl['OS']:.2f} °C\nST: {kpi_rl['ST']:.1f} dk"
if kpi_mpc:
    kpi_text += f"\n\n--- MPC METRİKLERİ ---\nMAE: {kpi_mpc['MAE']:.3f} °C\nOS: {kpi_mpc['OS']:.2f} °C\nST: {kpi_mpc['ST']:.1f} dk"

ax1.text(1.02, 0.5, kpi_text, transform=ax1.transAxes, verticalalignment='center',
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc'), fontsize=10, family='monospace')

ax1.set_title("Sıcaklık Kontrol Performansı: RL vs MPC", fontsize=14, fontweight='bold')
ax1.set_ylabel("Sıcaklık (°C)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- 4.2 YAKIT ANALİZİ ---
if df_mpc is not None:
    ax2.plot(time_mpc, df_mpc['fuel'], color='gray', alpha=0.2, label='MPC Yakıt')

ax2.plot(time_rl, fuel_rl, color='#FFBF00', lw=1.5, label='RL Yakıt (Hybrid)')
ax2.set_title("Aktüatör (Yakıt) Kullanımı", fontsize=12)
ax2.set_xlabel("Zaman (Dakika)")
ax2.set_ylabel("Yakıt Akışı")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.subplots_adjust(right=0.82) 
plt.show()

print(f"\nAnaliz Tamamlandı.")
print(f"RL MAE: {kpi_rl['MAE']:.4f} | MPC MAE: {kpi_mpc['MAE'] if kpi_mpc else 'N/A'}")
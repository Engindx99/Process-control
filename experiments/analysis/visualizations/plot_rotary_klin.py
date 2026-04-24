import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Import hatasını önlemek için proje kök dizinini ekliyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

try:
    from src.Rotaryklin.Rotary_klin import RotaryKilnPlant
except ImportError as e:
    print(f"Hata: Modül bulunamadı! Lütfen PYTHONPATH ayarını kontrol edin.\n{e}")
    sys.exit(1)

plant = RotaryKilnPlant()
steps = 21600 # 6 saat
fuel_val, fan_val = 18, 960

# --- HEDEF DEĞERLER (SETPOINTS) ---
setpoints = {
    "T1": 830.0,
    "T2": 1085.0,
    "T3": 1450.0, # Burning Zone hedefi
    "T4": 1000.0
}

records = []

print(f"--- Simülasyon Başlatıldı ({steps} adım) ---")

for i in range(steps):
    # Ağırlaştırılmış atalet sayesinde plant.step artık daha yavaş tepki verecek
    state = plant.step(fuel_cmd=fuel_val, fan_cmd=fan_val)
    state["step"] = i
    
    # Setpointleri kayıtlara ekle (Grafikte çizdirmek için)
    for k, v in setpoints.items():
        state[f"{k}_sp"] = v
        
    records.append(state)
    
    if i % (steps // 10) == 0:
        print(f"İlerleme: %{(i/steps)*100:.0f} (Adım: {i})")

print("--- Simülasyon Bitti. Grafik Çiziliyor... ---")
df = pd.DataFrame(records)

def plot_kiln_results(df):
    t = df["step"]
    last_step = t.iloc[-1]
    
    fig1, axs1 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    titles = ["Preheat", "Calcination", "Burning", "Cooling"]
    
    for idx, col in enumerate(["T1", "T2", "T3", "T4"]):
        # PV Çizimi
        axs1[idx].plot(t, df[col], lw=1.5, label="PV", color='darkred' if col=="T3" else 'tab:blue')
        
        # SP Çizimi
        sp_val = df[f"{col}_sp"].iloc[0]
        axs1[idx].axhline(y=sp_val, lw=1.2, ls='--', color='black', alpha=0.6)
        
        # --- ŞIK ÇÖZÜM: SAĞ KENAR ETİKETİ ---
        # Ekseni kalabalıklaştırmadan rakamı sağ boşluğa yazıyoruz
        axs1[idx].text(last_step * 1.01, sp_val, f'SP: {sp_val}°C', 
                       va='center', ha='left', fontsize=9, fontweight='bold',
                       color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
        
        axs1[idx].set_ylabel(f"{titles[idx]} (°C)")
        axs1[idx].grid(True, alpha=0.2)
        
        # Burning Zone özel limit koruması
        if col == "T3":
            axs1[idx].set_ylim(1430, 1470)

    # Sağ tarafta yazıların sığması için biraz boşluk bırak
    plt.subplots_adjust(right=0.88) 
    plt.show()

plot_kiln_results(df)
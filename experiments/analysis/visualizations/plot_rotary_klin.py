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
fuel_main, fuel_alt = 19.5, 2  # Deneme amaçlı biraz alternatif yakıt ekledik
fan_val = 960

# --- HEDEF DEĞERLER (SETPOINTS) ---
setpoints = {
    "T1": 830.0, "T2": 1085.0, "T3": 1450.0, "T4": 1000.0,
    "fCaO": 1.5  # Klinker kalite hedefi (Serbest Kireç %1.5)
}

records = []

print(f"--- Simülasyon Başlatıldı ({steps} adım) ---")

for i in range(steps):
    # Yeni step fonksiyonu imzasına uygun çağrı
    state = plant.step(fuel_main_cmd=fuel_main, fuel_alt_cmd=fuel_alt, fan_cmd=fan_val)
    state["step"] = i
    
    # Setpointleri kayıtlara ekle
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
    
    # Grafik sayısını 6'ya çıkardık (Sıcaklıklar + Kalite + Yakıt/LHV)
    fig, axs = plt.subplots(6, 1, figsize=(14, 15), sharex=True)
    plt.subplots_adjust(right=0.85, hspace=0.3)
    
    # 1-4: Sıcaklık Bölgeleri
    zones = [("T1", "Preheat", 700, 950), ("T2", "Calcination", 1000, 1150), 
             ("T3", "Burning", 1430, 1480), ("T4", "Cooling", 750, 1050)]
    
    for idx, (col, title, y_min, y_max) in enumerate(zones):
        axs[idx].plot(t, df[col], lw=1.5, color='darkred' if col=="T3" else 'tab:blue', label="Ölçülen")
        sp_val = df[f"{col}_sp"].iloc[0]
        axs[idx].axhline(y=sp_val, lw=1.2, ls='--', color='black', alpha=0.6)
        axs[idx].text(last_step * 1.01, sp_val, f'SP: {sp_val}°C', va='center', fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        axs[idx].set_ylabel(f"{title}\n(°C)")
        axs[idx].set_ylim(y_min, y_max)
        axs[idx].grid(True, alpha=0.2)

    # 5: KLİNKER KALİTE (Free Lime - fCaO)
    # Anlık fırın içindeki kaliteyi (açık renk) ve 20 dk sonra çıkacak olanı (koyu renk) çiziyoruz
    axs[4].plot(t, df["fCaO_instant"], lw=1, color='tab:orange', alpha=0.4, label="Fırın İçi (Anlık)")
    axs[4].plot(t, df["fCaO_delayed"], lw=2, color='tab:green', label="Laboratuvar (20 dk Gecikmeli)")
    axs[4].axhline(y=setpoints["fCaO"], lw=1.2, ls='--', color='black', alpha=0.6)
    axs[4].text(last_step * 1.01, setpoints["fCaO"], f'Hedef: %{setpoints["fCaO"]}', va='center', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    axs[4].set_ylabel("Free Lime\n(fCaO %)")
    axs[4].set_ylim(0, 4)
    axs[4].legend(loc='upper left', fontsize=8)
    axs[4].grid(True, alpha=0.2)

    # 6: YAKIT VE LHV DURUMU
    ax6_twin = axs[5].twinx()
    p1 = axs[5].plot(t, df["fuel_total"], lw=1.5, color='black', label="Toplam Yakıt")
    p2 = ax6_twin.plot(t, df["LHV_mix"], lw=1.5, color='purple', alpha=0.6, ls=':', label="Karma LHV")
    
    axs[5].set_ylabel("Yakıt (kg/s)")
    ax6_twin.set_ylabel("LHV (kcal/kg)")
    axs[5].set_xlabel("Zaman (Adım)")
    
    # Legend birleştirme
    lns = p1 + p2
    labs = [l.get_label() for l in lns]
    axs[5].legend(lns, labs, loc='upper left', fontsize=8)
    axs[5].grid(True, alpha=0.2)

    plt.suptitle(f"Döner Fırın Dijital İkiz Simülasyonu\n(LHV: {df['LHV_mix'].iloc[0]:.0f} kcal/kg | Verim: %11)", fontsize=14)
    plt.show()

plot_kiln_results(df)
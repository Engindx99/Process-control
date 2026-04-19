import matplotlib.pyplot as plt
import pandas as pd
from src.dt.dt import RotaryKilnPlant

# ==========================================
# UYUMLU CONFIGURATION (24 SAAT ANALİZİ)
# ==========================================
EPISODES = 1
STEP_DURATION = 5          # DİĞER KODLARLA UYUMLU: Her adım 5 saniye
TOTAL_MINUTES = 1440       # Hedeflenen toplam dakika (24 saat)
# UYUMLU ADIM SAYISI: (1440 dk * 60 sn) / 5 sn = 17280 adım
STEPS = int((TOTAL_MINUTES * 60) / STEP_DURATION) 

SETPOINT_TEMP = 1450
TARGET_O2 = 2.5

for ep in range(EPISODES):
    # 1. Veri Üretimi (Mevcut Plant yapına tam uyumlu)
    plant = RotaryKilnPlant(seed=None)  
    # 17280 adım çalıştırarak gerçek bir 24 saat simüle ediyoruz
    df = plant.run(steps=STEPS)

    # 2. Zamansal Dönüşüm (X ekseni 0'dan 1440 dakikaya akacak)
    t_min = (df["step"] * STEP_DURATION) / 60

    # =========================
    # TEMPERATURE PLOT (24 SAAT)
    # =========================
    plt.figure(figsize=(15, 6))
    plt.plot(t_min, df["temp"], color='tab:red', linewidth=0.8, label="Fırın Sıcaklığı")
    plt.axhline(y=SETPOINT_TEMP, color='black', linestyle='--', alpha=0.7, label=f"Hedef ({SETPOINT_TEMP}°C)")
    
    plt.title(f"24 Saatlik Sıcaklık Analizi ({TOTAL_MINUTES} Dakika) - Episode {ep+1}")
    plt.xlabel("Zaman (Dakika)")
    plt.ylabel("Sıcaklık (°C)")
    
    # Senin istediğin y-ekseni ayarı
    plt.yticks(range(1400, 1501, 25))
    plt.xlim(0, TOTAL_MINUTES)
    
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # =========================
    # O2 & PRESSURE PLOTS
    # =========================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # O2 Dinamiği
    ax1.plot(t_min, df["o2"], color='tab:blue', linewidth=0.8)
    ax1.axhline(y=TARGET_O2, color='green', linestyle=':', label="Hedef O2")
    ax1.set_ylabel("O2 (%)")
    ax1.set_title("24 Saatlik Oksijen ve Basınç Değişimi")
    ax1.grid(True, alpha=0.3)

    # Basınç Dinamiği
    ax2.plot(t_min, df["pressure"], color='tab:gray', linewidth=0.8)
    ax2.set_ylabel("Basınç (Pa)")
    ax2.set_xlabel("Zaman (Dakika)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
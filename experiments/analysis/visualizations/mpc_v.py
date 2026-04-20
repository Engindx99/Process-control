import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def compute_professional_kpis(df, setpoint=1450):
    """Ölçeklendirilmiş ve gürültüye dirençli kontrol metrikleri."""
    temp = df['temp'].values
    time = df['minutes'].values
    error = np.abs(temp - setpoint)
    
    # Settling Time için bandı 2.0 yapıyoruz (1448 - 1452 aralığı)
    # Bu, 0.46 MAE'li bir sistem için oturma bölgesini netleştirir.
    band = 2.0 
    within_band = error <= band
    settling_index = None
    
    for i in range(len(within_band) - 1, -1, -1):
        if not within_band[i]:
            settling_index = i + 1
            break
    
    if settling_index is not None and settling_index < len(time):
        settling_time = time[settling_index]
    elif all(within_band):
        settling_time = 0.0
    else:
        settling_time = np.nan

    overshoot = max(0, np.max(temp) - setpoint)
    total_mae = np.mean(error)

    return {
        "settling_time": settling_time, 
        "overshoot": overshoot, 
        "mae": total_mae,
        "band": band
    }

def plot_kiln_zoomed_triple(csv_path="data/simulation_results_v4.csv"):
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df['minutes'] = df['step'] * 5 / 60
    
    kpis = compute_professional_kpis(df)
    plt.style.use('seaborn-v0_8-muted')

    # --- PENCERE 1: SICAKLIK (1400-1500 ÖLÇEĞİ) ---
    plt.figure("Pencere 1: Termal Performans (Zoom)", figsize=(11, 6))
    plt.plot(df['minutes'], df['temp'], color='#E31A1C', lw=2, label='Fırın Sıcaklığı')
    plt.axhline(y=1450, color='black', linestyle='--', alpha=0.7, label='Hedef: 1450°C')
    
    # ÖLÇEKLENDİRME: İstediğin 1400-1500 aralığı
    plt.ylim(1440, 1460)
    
    # KPI Metin Kutusu
    kpi_text = (f"MAE: {kpis['mae']:.4f}\n"
                f"Overshoot: {kpis['overshoot']:.2f}°C\n"
                f"Settling Time: {kpis['settling_time']:.2f} dk\n"
                f"Analiz Bandı: ±{kpis['band']}°C")
    
    plt.gca().text(0.02, 0.95, kpi_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title("Fırın Sıcaklık Analizi (Ölçek: 1400-1500°C)", fontsize=13, fontweight='bold')
    plt.xlabel("Zaman (Dakika)")
    plt.ylabel("Sıcaklık (°C)")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')

    # --- PENCERE 2: O2 ---
    plt.figure("Pencere 2: O2 Seviyesi", figsize=(11, 5))
    plt.plot(df['minutes'], df['o2'], color='#1F78B4', lw=1.5)
    plt.axhline(y=2.25, color='blue', linestyle=':', alpha=0.5)
    plt.title("O2 Konsantrasyonu (%)", fontsize=13, fontweight='bold')
    plt.ylabel("O2 %")
    plt.grid(True, alpha=0.3)

    # --- PENCERE 3: AKTÜATÖRLER ---
    plt.figure("Pencere 3: Kontrol Çıktıları", figsize=(11, 5))
    ax3 = plt.gca()
    ax3_twin = ax3.twinx()
    ax3.step(df['minutes'], df['fuel'], color='#FF7F00', lw=2, label='Yakıt')
    ax3_twin.step(df['minutes'], df['fan'], color='#444444', lw=1.5, label='Fan', alpha=0.6)
    plt.title("Yakıt ve Fan Kararları", fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    plot_kiln_zoomed_triple()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def compute_professional_kpis(df, setpoint=1450):
    temp = df['temp'].values
    # Saniyeyi dakikaya çeviriyoruz
    time = df['second'].values / 60 
    error = np.abs(temp - setpoint)
    
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

def plot_kiln_final_results(csv_path="kiln_v10_12_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı! Lütfen simülasyonu önce çalıştırın.")
        return

    # Veriyi yükle
    df = pd.read_csv(csv_path)
    # Zaman eksenini saniyeden dakikaya çevir
    df['minutes'] = df['second'] / 60
    
    kpis = compute_professional_kpis(df)
    plt.style.use('seaborn-v0_8-muted')

    # --- PENCERE 1: SICAKLIK ---
    plt.figure("Pencere 1: Termal Performans", figsize=(12, 6))
    plt.plot(df['minutes'], df['temp'], color='#E31A1C', lw=1.2, label='Fırın Sıcaklığı')
    plt.axhline(y=1450, color='black', linestyle='--', alpha=0.7, label='Hedef: 1450°C')
    
    # Zoom ayarı: 1440-1460 arası (v10.12 performansı için ideal)
    plt.ylim(1440, 1460)
    
    kpi_text = (f"MAE: {kpis['mae']:.4f}\n"
                f"Overshoot: {kpis['overshoot']:.2f}°C\n"
                f"Settling Time: {kpis['settling_time']:.2f} dk\n"
                f"Analiz Bandı: ±{kpis['band']}°C")
    
    plt.gca().text(0.02, 0.95, kpi_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.title(f"Fırın Sıcaklık Analizi - v10.12 (Kalman Entegre)", fontsize=14)
    plt.xlabel("Zaman (Dakika)")
    plt.ylabel("Sıcaklık (°C)")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')

    # --- PENCERE 2: O2 VE BASINÇ (Filtre Etkisi) ---
    plt.figure("Pencere 2: Gaz ve Basınç Dengesi", figsize=(12, 6))
    ax2 = plt.gca()
    # Ham O2 (silik) ve Filtreli O2 (net)
    ax2.plot(df['minutes'], df['o2'], color='blue', alpha=0.2, label='Ham O2')
    ax2.plot(df['minutes'], df['filt_o2'], color='blue', lw=1.5, label='Kalman Filtreli O2')
    ax2.set_ylabel("O2 %", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Sağ eksen: Basınç
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['minutes'], df['filt_press'], color='green', lw=1, label='Filtreli Basınç')
    ax2_twin.set_ylabel("Basınç (Pa)", color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    
    plt.title("Oksijen ve Basınç Kararlılığı", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # --- PENCERE 3: AKTÜATÖRLER ---
    plt.figure("Pencere 3: Kontrol Çıktıları", figsize=(12, 5))
    ax3 = plt.gca()
    ax3_twin = ax3.twinx()
    
    ax3.step(df['minutes'], df['fuel'], color='#FF7F00', lw=2, label='Yakıt (Fuel)')
    ax3_twin.step(df['minutes'], df['fan'], color='#444444', lw=1.5, label='Fan Speed', alpha=0.6)
    
    ax3.set_xlabel("Zaman (Dakika)")
    ax3.set_ylabel("Yakıt Akışı", color='#FF7F00')
    ax3_twin.set_ylabel("Fan Devri", color='#444444')
    
    plt.title("MPC Aktüatör Kararları", fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_kiln_final_results()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_kiln_fixed_1450(csv_path="data/pure_mpc_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return

    df = pd.read_csv(csv_path)
    # Kolon isimlerini standartlaştır
    df.columns = [c.lower() for c in df.columns]
    
    # Gerekli kolonları eşle
    t_col = 'temperature' if 'temperature' in df.columns else 'temp'
    
    # Zaman hesapla (5 sn adımlar)
    df['minutes'] = df['step'] * 5 / 60

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # --- 1. GRAFİK: SICAKLIK (1440-1460 Odaklı) ---
    ax1.plot(df['minutes'], df[t_col], color='red', label='Fırın Sıcaklığı', linewidth=1.2)
    
    # Tek bir sabit Setpoint çizgisi
    ax1.axhline(y=1450, color='blue', linestyle='--', label='Hedef (1450°C)', alpha=0.3)
    
    # Eksen Limitleri Ayarı (İstediğin aralık)
    ax1.set_ylim(1445, 1455) 
    
    ax1.set_ylabel("Sıcaklık (°C)")
    ax1.set_title("Döner Fırın Kararlılık Analizi (Sabit 1450°C)")
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    # İstatistik kutusu (Sadece 1450'ye göre hata)
    mae = np.mean(np.abs(df[t_col] - 1450))
    std = np.std(df[t_col])
    stats_text = f"MAE: {mae:.4f} °C\nSTD: {std:.4f} °C"
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- 2. GRAFİK: YAKIT ---
    ax2.plot(df['minutes'], df['fuel'], color='goldenrod', label='Yakıt Tüketimi')
    ax2.axhline(y=28.0, color='darkred', linestyle='--', label='Maks. Limit (28.0)')
    
    # Yakıt grafiği alt sınırı (Daha net görmek için)
    ax2.set_ylim(10, 30)
    
    ax2.set_ylabel("Yakıt (kg/h)")
    ax2.set_xlabel("Zaman (Dakika)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    print("Grafik güncellendi: 1440-1460 aralığına odaklanıldı.")
    plt.show()

if __name__ == "__main__":
    plot_kiln_fixed_1450()
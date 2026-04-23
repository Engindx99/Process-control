import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_kiln_smooth_results(csv_path="kiln_v10_12_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return

    df = pd.read_csv(csv_path)
    df['minutes'] = df['second'] / 60
    
    # 1. YAKIT YUMUŞATMA (Merdiven görüntüsünü siler)
    # 10 saniyelik bir pencere ile hareketli ortalama alıyoruz
    df['fuel_smooth'] = df['fuel'].rolling(window=10, center=True).mean()
    df['fan_smooth'] = df['fan'].rolling(window=10, center=True).mean()

    plt.style.use('seaborn-v0_8-muted')

    # --- PENCERE 1: SICAKLIK ---
    plt.figure("Sıcaklık Analizi", figsize=(12, 5))
    plt.plot(df['minutes'], df['temp'], color='#E31A1C', lw=1, label='Fırın Sıcaklığı')
    plt.axhline(y=1450, color='black', linestyle='--', alpha=0.6)
    plt.ylim(1440, 1460)
    plt.title("Sıcaklık Kararlılığı (v10.12)")
    plt.grid(True, alpha=0.2)

    # --- PENCERE 2: OKSİJEN ---
    plt.figure("Oksijen Analizi", figsize=(12, 5))
    plt.plot(df['minutes'], df['filt_o2'], color='blue', lw=1.2, label='Filtreli O2')
    plt.title("Oksijen Seviyesi (%)")
    plt.grid(True, alpha=0.2)

    # --- PENCERE 3: AKTÜATÖRLER (Smooth & No-Step) ---
    plt.figure("Aktüatör Kararları (Yumuşatılmış)", figsize=(12, 6))
    ax3 = plt.gca()
    ax3_twin = ax3.twinx()
    
    # 'plt.plot' kullanarak noktaları düz birleştiriyoruz (Merdiven yok)
    # 10 saniyede bir örnek alarak (iloc[::10]) çizgi karmaşasını engelliyoruz
    df_plot = df.iloc[::10, :] 
    
    ax3.plot(df_plot['minutes'], df_plot['fuel_smooth'], color='#FF7F00', lw=2, label='Yakıt (Trend)')
    ax3_twin.plot(df_plot['minutes'], df_plot['fan_smooth'], color='#444444', lw=1.5, alpha=0.5, label='Fan (Trend)')
    
    ax3.set_xlabel("Zaman (Dakika)")
    ax3.set_ylabel("Yakıt Akışı", color='#FF7F00')
    ax3_twin.set_ylabel("Fan Devri", color='#444444')
    
    plt.title("Yumuşatılmış Yakıt ve Fan Hareketleri (10s Trend)")
    ax3.grid(True, alpha=0.3)
    
    # Legendları birleştirme
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_kiln_smooth_results()
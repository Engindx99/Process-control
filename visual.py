import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_6h_analysis(csv_path="mpc_optim.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Hata: {csv_path} bulunamadı. Önce simülasyonu çalıştırmalısın.")
        return

    t_min = df["second"] / 60
    setpoint = 1450
    mae = df['error'].mean()

    # 3'lü Alt Alta Grafik Yapısı
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"6 Saatlik Dijital İkiz Performans Analizi (Final MAE: {mae:.4f})", fontsize=16)

    # 1. SICAKLIK GRAFİĞİ
    ax1.plot(t_min, df["temp"], color='#d62728', linewidth=1, label="Fırın Sıcaklığı")
    ax1.axhline(y=setpoint, color='black', linestyle='--', alpha=0.8, label=f"Hedef ({setpoint}°C)")
    ax1.fill_between(t_min, setpoint - 5, setpoint + 5, color='gray', alpha=0.2, label="±5°C Tolerans")
    ax1.set_ylabel("Sıcaklık (°C)")
    ax1.set_ylim(1420, 1470) # Hataları daha net görmek için daraltılmış eksen
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # 2. YAKIT (FUEL) GRAFİĞİ - Sorunun kaynağı burası olabilir
    ax2.step(t_min, df["fuel"], color='#ff7f0e', where='post', linewidth=1.2, label="MPC Yakıt Komutu")
    ax2.set_ylabel("Yakıt Miktarı")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # 3. OKSİJEN (O2) GRAFİĞİ
    ax3.plot(t_min, df["o2"], color='#1f77b4', linewidth=1, label="O2 Seviyesi")
    ax3.axhline(y=2.2, color='green', linestyle=':', label="Hedef O2 (2.2%)")
    ax3.set_ylabel("O2 (%)")
    ax3.set_xlabel("Zaman (Dakika)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Grafiği Kaydet ve Göster
    plt.savefig("performance_analysis.png", dpi=300)
    print("📊 Grafik 'performance_analysis.png' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    visualize_6h_analysis()
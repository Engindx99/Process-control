import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_integrated_hybrid_clean(csv_path="experiments/plots/training_results.csv"):
    # 1. Veri Kontrolü
    if not os.path.exists(csv_path):
        print(f"❌ Hata: {csv_path} bulunamadı. Lütfen önce main.py çalıştırın.")
        return

    df = pd.read_csv(csv_path)
    
    # 2. Veri Hazırlığı
    df['fuel_smooth'] = df['fuel'].rolling(window=10, min_periods=1).mean()
    df['error'] = df['temp'] - 1450 # Setpoint hatası
    
    # 3. Görselleştirme Ayarları
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- ÜST PANEL: SICAKLIK TAKİBİ (Saf MPC Stilinde) ---
    ax1.plot(df['step'], df['temp'], label='Fırın Sıcaklığı (°C)', color='#1f77b4', linewidth=1.5)
    ax1.axhline(y=1450, color='red', linestyle='--', alpha=0.7, label='Hedef (1450°C)')
    
    # Hedef ve gerçek arasındaki bölgeyi renklendirme (Saf MPC'deki görsel efekt)
    ax1.fill_between(df['step'], df['temp'], 1450, color='red', alpha=0.05)
    
    ax1.set_ylabel("Sıcaklık (°C)", fontsize=12)
    ax1.set_title("Döner Fırın: Hibrit Kontrol Performansı", fontsize=15, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # --- ALT PANEL: YAKIT AKIŞI ---
    ax2.plot(df['step'], df['fuel_smooth'], label='Yakıt Akışı (m³/h)', color='gold', linewidth=1.5)
    
    # Dinamik Y-Ekseni (2'şerli artış)
    y_min_f = int(np.floor(df['fuel'].min() / 2) * 2) - 2
    y_max_f = int(np.ceil(df['fuel'].max() / 2) * 2) + 2
    ax2.set_yticks(np.arange(y_min_f, y_max_f + 1, 2)) 
    
    ax2.set_ylabel("Yakıt (m³/h)", fontsize=12)
    ax2.set_xlabel("Zaman Adımı (Step)", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # X-Ekseni İşaretçileri (3000 adım için 250'şer aralık)
    ax2.set_xticks(np.arange(0, df['step'].max() + 1, 250))

    # --- PERFORMANS İSTATİSTİKLERİ (Saf MPC Kodundaki Mantık) ---
    mae = df['error'].abs().mean()
    std_dev = df['temp'].std()
    stats_text = f"Hibrit Sistem Özeti:\nMAE: {mae:.2f}°C\nStd Dev: {std_dev:.2f}"
    
    # İstatistik kutusunu sol alt köşeye ekle
    plt.gcf().text(0.13, 0.05, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', edgecolor='#1f77b4', boxstyle='round,pad=0.5', alpha=0.8))

    plt.tight_layout()
    
    # Kaydet ve Göster
    output_path = "experiments/plots/hybrid_clean_performance.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Sadeleştirilmiş hibrit grafik kaydedildi: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_integrated_hybrid_clean()
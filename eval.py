import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_clean_fuel_performance(csv_path="data/training_results.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ Hata: {csv_path} bulunamadı.")
        return

    df = pd.read_csv(csv_path)
    
    # 5 adımlık hareketli ortalama ile çizgiyi biraz daha toplu gösterelim
    df['fuel_smooth'] = df['fuel'].rolling(window=5, min_periods=1).mean()
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 1. SICAKLIK GRAFİĞİ
    axes[0].plot(df['step'], df['temp'], color='#2c7bb6', linewidth=1.3, label='Fırın Sıcaklığı')
    axes[0].axhline(1450, color='#d7191c', linestyle='--', label='Hedef 1450°C', alpha=0.8)
    axes[0].set_ylabel("Sıcaklık (°C)", fontsize=12)
    axes[0].set_title("Döner Fırın Sıcaklık Kontrolü", fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')

    # 2. YAKIT GRAFİĞİ (SADE ÇİZGİ VE 2'ŞERLİ EKSEN)
    # Alt dolguyu (fill_between) tamamen kaldırdık
    axes[1].plot(df['step'], df['fuel_smooth'], color='#f58518', linewidth=1.3, label='Toplam Yakıt Debisi')
    
    # Y-Ekseni Kademelerini 2'şer birim aralıkla ayarla
    y_min = int(np.floor(df['fuel'].min() / 2) * 2) - 2
    y_max = int(np.ceil(df['fuel'].max() / 2) * 2) + 2
    axes[1].set_yticks(np.arange(y_min, y_max + 1, 2)) 
    
    axes[1].set_ylabel("Yakıt (m³/h)", fontsize=12)
    axes[1].set_xlabel("Zaman Adımı (Step)", fontsize=12)
    axes[1].set_title("Yakıt Tüketimi", fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    
    # Kaydet ve Göster
    output_path = "data/clean_fuel_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Sadeleştirilmiş grafik kaydedildi: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_clean_fuel_performance()
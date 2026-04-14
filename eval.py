import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_mpc_vs_hybrid_comparison(pure_csv="data/pure_mpc_results.csv", hybrid_csv="data/training_results.csv"):
    # 1. Veri Kontrolü
    if not os.path.exists(pure_csv) or not os.path.exists(hybrid_csv):
        print("❌ Hata: Dosyalardan biri eksik!")
        print(f"Aranan yollar:\n1. {pure_csv}\n2. {hybrid_csv}")
        return

    # 2. Verileri Yükle
    df_pure = pd.read_csv(pure_csv)
    df_hybrid = pd.read_csv(hybrid_csv)

    # 3. Görselleştirme Ayarları
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # --- ÜST PANEL: SICAKLIK KIYASLAMASI ---
    ax1.plot(df_pure['step'], df_pure['temp'], label='Saf MPC (Geleneksel)', color='#95a5a6', alpha=0.6, linestyle='--')
    ax1.plot(df_hybrid['step'], df_hybrid['temp'], label='Hibrit (MPC + RL)', color='#1f77b4', linewidth=1.5)
    
    ax1.axhline(y=1450, color='red', linestyle='-', alpha=0.5, label='Hedef 1450°C')
    ax1.set_ylabel("Sıcaklık (°C)", fontsize=12)
    ax1.set_title("Sıcaklık Kontrolü: Saf MPC vs Hibrit Zeka", fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right')

    # --- ALT PANEL: YAKIT TÜKETİMİ KIYASLAMASI ---
    # Hareketli ortalama ile daha temiz bir kıyaslama
    pure_fuel_smooth = df_pure['fuel'].rolling(window=15, min_periods=1).mean()
    hybrid_fuel_smooth = df_hybrid['fuel'].rolling(window=15, min_periods=1).mean()

    ax2.plot(df_pure['step'], pure_fuel_smooth, label='Saf MPC Yakıt', color='#e74c3c', alpha=0.5, linestyle='--')
    ax2.plot(df_hybrid['step'], hybrid_fuel_smooth, label='Hibrit Yakıt', color='#f39c12', linewidth=1.5)
    
    ax2.set_ylabel("Yakıt Tüketimi (m³/h)", fontsize=12)
    ax2.set_xlabel("Zaman Adımı (Step)", fontsize=12)
    ax2.set_title("Yakıt Verimliliği Kıyaslaması", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')

    # X-Ekseni İşaretçileri (3000 adım için)
    ax2.set_xticks(np.arange(0, 3001, 250))

    # --- PERFORMANS KARŞILAŞTIRMA TABLOSU (Metin Kutusu) ---
    pure_mae = (df_pure['temp'] - 1450).abs().mean()
    hybrid_mae = (df_hybrid['temp'] - 1450).abs().mean()
    
    improvement = ((pure_mae - hybrid_mae) / pure_mae) * 100

    stats_text = (f"📊 Kıyaslama Özeti (3000 Adım):\n"
                  f"---------------------------\n"
                  f"Saf MPC MAE:   {pure_mae:.2f}°C\n"
                  f"Hibrit MAE:    {hybrid_mae:.2f}°C\n"
                  f"İyileşme:      %{improvement:.1f}")

    plt.gcf().text(0.13, 0.05, stats_text, fontsize=11, fontfamily='monospace',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9))

    plt.tight_layout()
    
    output_path = "experiments/plots/mpc_vs_hybrid_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Karşılaştırma grafiği hazır: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_mpc_vs_hybrid_comparison()
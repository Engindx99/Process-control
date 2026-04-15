import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_mpc_vs_hybrid_comparison(pure_csv="data/pure_mpc_results.csv", hybrid_csv="data/training_results.csv"):
    # 1. Veri Kontrolü
    if not os.path.exists(pure_csv) or not os.path.exists(hybrid_csv):
        print("❌ Hata: Dosyalardan biri eksik!")
        return

    # 2. Verileri Yükle
    df_pure = pd.read_csv(pure_csv)
    df_hybrid = pd.read_csv(hybrid_csv)

    # 3. GÖRSELLEŞTİRME - PENCERE 1: GENEL PERFORMANS (Sıcaklık ve Yakıt)
    sns.set_theme(style="whitegrid")
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- ÜST PANEL: SICAKLIK ---
    ax1.plot(df_pure['step'], df_pure['temp'], label='Saf MPC', color='#95a5a6', alpha=0.5, linestyle='--')
    ax1.plot(df_hybrid['step'], df_hybrid['temp'], label='Hibrit (MPC + RL)', color='#1f77b4', linewidth=1.5)
    ax1.axhline(y=1450, color='red', linestyle='-', alpha=0.5, label='Hedef 1450°C')
    ax1.set_ylabel("Sıcaklık (°C)", fontsize=12)
    ax1.set_title("Sıcaklık Kontrolü: Saf MPC vs Hibrit Zeka (0.05 Mass)", fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right')

    # --- ALT PANEL: YAKIT ---
    pure_fuel_smooth = df_pure['fuel'].rolling(window=20, min_periods=1).mean()
    hybrid_fuel_smooth = df_hybrid['fuel'].rolling(window=20, min_periods=1).mean()
    ax2.plot(df_pure['step'], pure_fuel_smooth, label='Saf MPC Yakıt', color='#e74c3c', alpha=0.4, linestyle='--')
    ax2.plot(df_hybrid['step'], hybrid_fuel_smooth, label='Hibrit Yakıt', color='#f39c12', linewidth=1.5)
    ax2.set_ylabel("Yakıt Tüketimi (m³/h)", fontsize=12)
    ax2.set_xlabel("Zaman Adımı (Step)", fontsize=12)
    ax2.legend(loc='upper right')

    # İstatistik Kutusu
    pure_mae = (df_pure['temp'] - 1450).abs().mean()
    hybrid_mae = (df_hybrid['temp'] - 1450).abs().mean()
    improvement = ((pure_mae - hybrid_mae) / pure_mae) * 100
    stats_text = (f"📊 MAE Özet:\nMPC: {pure_mae:.2f}°C\nHibrit: {hybrid_mae:.2f}°C\nİyileşme: %{improvement:.1f}")
    ax2.text(0.02, 0.05, stats_text, transform=ax2.transAxes, fontsize=10, fontfamily='monospace', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    plt.tight_layout()
    print("📈 İlk pencere açıldı. Analize devam etmek için pencereyi kapatın...")
    plt.show() # İlk pencere burada bloklar, kapanınca kod devam eder.

    # 4. GÖRSELLEŞTİRME - PENCERE 2: VERİMLİLİK VE O2 ANALİZİ
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- ÜST PANEL: OKSİJEN (O2) ---
    ax3.plot(df_pure['step'], df_pure['o2'], label='Saf MPC O2', color='#95a5a6', alpha=0.5, linestyle='--')
    ax3.plot(df_hybrid['step'], df_hybrid['o2'], label='Hibrit O2', color='#2ca02c', linewidth=1.5)
    ax3.axhline(y=3.0, color='#d62728', linestyle=':', label='İdeal Nokta (%3)')
    
    # Verimli bölgeyi boyayalım (Fiziksel gerçeğimiz: %2 - %4.5 arası)
    ax3.fill_between(df_hybrid['step'], 2.0, 4.5, color='green', alpha=0.1, label='Yüksek Verim Bölgesi')
    ax3.set_ylabel("Oksijen (%)", fontsize=12)
    ax3.set_title("O2 Yönetimi ve Yanma Kalitesi", fontsize=15, fontweight='bold')
    ax3.legend(loc='upper right')

    # --- ALT PANEL: VERİMLİLİK (EFFICIENCY) ---
    if 'efficiency' in df_hybrid.columns:
        ax4.plot(df_pure['step'], df_pure['efficiency'], label='Saf MPC Verim', color='#7f8c8d', alpha=0.4)
        ax4.plot(df_hybrid['step'], df_hybrid['efficiency'], label='Hibrit Verim', color='#9467bd', linewidth=2)
        ax4.set_ylabel("Sistem Verimliliği (0-1)", fontsize=12)
        ax4.set_title("Toplam Sistem Verimlilik Kıyaslaması", fontsize=14)
        ax4.legend(loc='lower right')

    ax4.set_xlabel("Zaman Adımı (Step)", fontsize=12)
    plt.tight_layout()
    
    # Kaydetme ve Gösterme
    if not os.path.exists("experiments/plots"): os.makedirs("experiments/plots")
    plt.savefig("experiments/plots/efficiency_analysis.png", dpi=300)
    print("✅ İkinci analiz grafiği hazır.")
    plt.show()

if __name__ == "__main__":
    plot_mpc_vs_hybrid_comparison()
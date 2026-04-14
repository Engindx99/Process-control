import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pure_mpc_results(csv_path="data/pure_mpc_results.csv"):
    # 1. Veriyi Yükle
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Hata: {csv_path} bulunamadı. Önce testi çalıştırın.")
        return

    # Görselleştirme stilini ayarla
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- ÜST PANEL: SICAKLIK TAKİBİ ---
    ax1.plot(df['step'], df['temp'], label='Fırın Sıcaklığı (°C)', color='#1f77b4', linewidth=1.2)
    ax1.axhline(y=1450, color='red', linestyle='--', alpha=0.7, label='Setpoint (1450°C)')
    
    # Başlangıç ve Setpoint bölgelerini renklendir
    ax1.fill_between(df['step'], df['temp'], 1450, color='red', alpha=0.05)
    
    ax1.set_ylabel("Sıcaklık (°C)", fontsize=12)
    ax1.set_title("Döner Fırın: Saf MPC Kontrol Performansı", fontsize=15, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # --- ALT PANEL: YAKIT AKIŞI ---
    ax2.plot(df['step'], df['fuel'], label='Yakıt Akışı (m³/h)', color='#2ca02c', linewidth=1.2)
    
    ax2.set_ylabel("Yakıt (m³/h)", fontsize=12)
    ax2.set_xlabel("Zaman Adımı (Step)", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # İstatistikleri Grafiğe Ekle (Sol Alt Köşe)
    mae = df['error'].abs().mean()
    std_dev = df['temp'].std()
    stats_text = f"Performans Özeti:\nMAE: {mae:.2f}\nStd Dev: {std_dev:.2f}"
    plt.gcf().text(0.13, 0.05, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_pure_mpc_results()
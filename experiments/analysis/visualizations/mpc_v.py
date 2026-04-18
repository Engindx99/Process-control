import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_kiln_custom_windows(csv_path):
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return

    df = pd.read_csv(csv_path)
    plt.style.use('seaborn-v0_8-muted')

    # --- PENCERE 1: ANA KONTROL DÖNGÜSÜ (Sıcaklık ve Yakıt) ---
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Sıcaklık Paneli
    axes1[0].plot(df['step'], df['temp'], label='Fırın Sıcaklığı (°C)', color='red', lw=1.2)
    axes1[0].axhline(y=1450, color='red', linestyle='--', alpha=0.6, label='Hedef (1450°C)')
    axes1[0].set_title(f"Sıcaklık ve Yakıt İlişkisi\nMAE: {df['error'].abs().mean():.4f}°C", fontsize=14)
    axes1[0].set_ylabel("Sıcaklık (°C)")
    axes1[0].legend(loc='upper right')
    axes1[0].grid(True, alpha=0.3)

    # Yakıt Paneli
    axes1[1].plot(df['step'], df['fuel'], label='Yakıt Akışı (fuel)', color='#DAA520', lw=1.2)
    axes1[1].set_ylabel("Yakıt Miktarı")
    axes1[1].set_xlabel("Adım (Step)")
    axes1[1].legend(loc='upper right')
    axes1[1].grid(True, alpha=0.3)
    
    fig1.tight_layout()
    fig1.canvas.manager.set_window_title('Sıcaklık & Yakıt Analizi')

    # --- PENCERE 2: YANMA VE HAVA ANALİZİ (Fan, O2 ve Hata) ---
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Fan ve O2 Paneli (Twin Axis)
    ax2_twin = axes2[0].twinx()
    p1, = axes2[0].plot(df['step'], df['fan'], label='Fan Hızı (RPM)', color='#9467bd', lw=1.2)
    p2, = ax2_twin.plot(df['step'], df['o2'], label='O2 %', color='#2ca02c', lw=1.2, linestyle='-.')
    
    axes2[0].set_title("Hava ve Gaz Dengesi", fontsize=12)
    axes2[0].set_ylabel("Fan RPM")
    ax2_twin.set_ylabel("O2 %")
    axes2[0].legend(handles=[p1, p2], loc='upper right')
    axes2[0].grid(True, alpha=0.3)

    # Anlık Hata Paneli
    axes2[1].fill_between(df['step'], df['error'], 0, where=(df['error'] >= 0), color='green', alpha=0.3)
    axes2[1].fill_between(df['step'], df['error'], 0, where=(df['error'] < 0), color='red', alpha=0.3)
    axes2[1].plot(df['step'], df['error'], color='black', lw=0.8, alpha=0.5, label='Hata (Error)')
    axes2[1].set_ylabel("Hata (°C)")
    axes2[1].set_xlabel("Adım (Step)")
    axes2[1].legend(loc='upper right')
    axes2[1].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.canvas.manager.set_window_title('Hava Dengesi & Hata Dağılımı')

    plt.show()

if __name__ == "__main__":
    plot_kiln_custom_windows("data/pure_mpc_results.csv")
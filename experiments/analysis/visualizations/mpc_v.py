import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def compute_professional_kpis(df, setpoint=1450, band=2):
    """Gelişmiş kontrol metrikleri hesaplar."""
    # Kolon isimlerini küçük harfe çevir ve kontrol et
    df.columns = [c.lower() for c in df.columns]
    
    # Senin Plant sınıfından gelen 'temp' kolonu öncelikli
    t_col = 'temp' if 'temp' in df.columns else ('temperature' if 'temperature' in df.columns else None)
    
    if t_col is None:
        return None

    temp = df[t_col].values
    # Step'ten dakikaya çevrim (Step aralığın 5 sn ise /12, 60 sn ise direkt step)
    # config'deki step_duration_sec'e göre burayı revize edebilirsin
    time = df['minutes'].values 
    error = temp - setpoint

    # 1. Settling Time (Hata bandının içine girip bir daha çıkmadığı an)
    within_band = np.abs(error) <= band
    settling_index = None
    for i in range(len(within_band)):
        if all(within_band[i:]):
            settling_index = i
            break
    settling_time = time[settling_index] if settling_index is not None else np.nan

    # 2. Overshoot (Aşım)
    overshoot = np.max(temp) - setpoint
    overshoot = max(0, overshoot)

    # 3. Rise Time (%10 -> %90)
    try:
        t_low, t_high = setpoint * 0.1, setpoint * 0.9
        t1 = time[np.where(temp >= t_low)[0][0]]
        t2 = time[np.where(temp >= t_high)[0][0]]
        rise_time = t2 - t1
    except:
        rise_time = np.nan

    # 4. Steady State Hassasiyeti
    if settling_index is not None:
        ss_error = error[settling_index:]
        ss_mae = np.mean(np.abs(ss_error))
        ss_std = np.std(ss_error)
    else:
        ss_mae, ss_std = np.nan, np.nan

    return {
        "settling_time": settling_time,
        "overshoot": overshoot,
        "rise_time": rise_time,
        "ss_mae": ss_mae,
        "ss_std": ss_std
    }

def plot_kiln_custom_windows(csv_path="data/mpc_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı! Lütfen önce main.py'yi çalıştırın.")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    
    # Eksik kolonları tamamla
    if 'temp' not in df.columns and 'temperature' in df.columns:
        df['temp'] = df['temperature']
    
    # Zaman ekseni hesabı (5 saniyelik adımlar için dakikaya çevrim)
    # Not: Eğer adımın 1 dk ise sadece df['step'] kullanabilirsin.
    df['minutes'] = df['step'] * 5 / 60 
    
    kpis = compute_professional_kpis(df)
    if kpis is None:
        print("Sıcaklık verisi bulunamadı!")
        return

    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- 1. GRAFİK: SICAKLIK ---
    ax1.plot(df['minutes'], df['temp'], color='red', lw=1.2, label='Fırın Sıcaklığı')
    ax1.axhline(y=1450, color='blue', linestyle='--', alpha=0.6, label='Hedef (1450°C)')
    
    # Görünümü iyileştirmek için dinamik limit (Hedef ± 20 derece)
    ax1.set_ylim(1430, 1470) 

    kpi_text = (
        f"Rise Time: {kpis['rise_time']:.2f} dk\n"
        f"Settling Time: {kpis['settling_time']:.2f} dk\n"
        f"Overshoot: {kpis['overshoot']:.2f} °C\n"
        f"SS MAE: {kpis['ss_mae']:.4f} °C\n"
        f"SS STD: {kpis['ss_std']:.4f} °C"
    )
    ax1.text(0.02, 0.95, kpi_text, transform=ax1.transAxes, verticalalignment='top',
             fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

    ax1.set_title("Rotary Kiln - MPC Kontrol Performans Analizi", fontsize=14)
    ax1.set_ylabel("Sıcaklık (°C)", fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')

    # --- 2. GRAFİK: YAKIT ---
    ax2.plot(df['minutes'], df['fuel'], color='#DAA520', lw=1.2, label='Yakıt (Fuel Command)')
    
    fuel_min, fuel_max = df['fuel'].min(), df['fuel'].max()
    ax2.set_ylim(max(0, fuel_min - 2), fuel_max + 2)
    
    ax2.set_ylabel("Yakıt (kg/h)", fontsize=12)
    ax2.set_xlabel("Zaman (Dakika)", fontsize=12)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    # Grafiği kaydetmek istersen: plt.savefig("data/mpc_performance.png")
    plt.show()

if __name__ == "__main__":
    plot_kiln_custom_windows()
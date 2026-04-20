import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def compute_professional_kpis(df, setpoint=1450, band=2):
    """Gelişmiş kontrol metrikleri hesaplar."""
    df.columns = [c.lower() for c in df.columns]
    
    t_col = 'temp' if 'temp' in df.columns else ('temperature' if 'temperature' in df.columns else None)
    if t_col is None: return None

    temp = df[t_col].values
    time = df['minutes'].values 
    error = temp - setpoint

    # 1. Settling Time
    within_band = np.abs(error) <= band
    settling_index = None
    for i in range(len(within_band)):
        if all(within_band[i:]):
            settling_index = i
            break
    settling_time = time[settling_index] if settling_index is not None else np.nan

    # 2. Overshoot
    overshoot = max(0, np.max(temp) - setpoint)

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
        print(f"Hata: {csv_path} bulunamadı!")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    
    if 'temp' not in df.columns and 'temperature' in df.columns:
        df['temp'] = df['temperature']
    
    # Zaman ekseni hesabı (5 saniyelik adımlar)
    df['minutes'] = df['step'] * 5 / 60 
    
    kpis = compute_professional_kpis(df)
    
    # Görselleştirme Stili
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # --- 1. PANEL: SICAKLIK (TEMP) ---
    ax1 = axes[0]
    ax1.plot(df['minutes'], df['temp'], color='#E31A1C', lw=1.5, label='Fırın Sıcaklığı (°C)')
    ax1.axhline(y=1450, color='black', linestyle='--', alpha=0.5, label='Set Point')
    ax1.set_ylim(1430, 1475)
    ax1.set_ylabel("Sıcaklık", fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # KPI Box
    if kpis:
        kpi_text = (f"Rise Time: {kpis['rise_time']:.1f}m | Settling: {kpis['settling_time']:.1f}m | "
                    f"Overshoot: {kpis['overshoot']:.1f}°C | MAE: {kpis['ss_mae']:.3f}")
        ax1.set_title(f"Rotary Kiln Kontrol Analizi\n{kpi_text}", fontsize=12, pad=15)

    # --- 2. PANEL: EMİSYONLAR (O2 & CO2) ---
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    p1, = ax2.plot(df['minutes'], df['o2'], color='#1F78B4', label='O2 (%)')
    p2, = ax2_twin.plot(df['minutes'], df['co2'], color='#33A02C', label='CO2 (%)', alpha=0.7)
    ax2.set_ylabel("O2 (%)", color='#1F78B4', fontweight='bold')
    ax2_twin.set_ylabel("CO2 (%)", color='#33A02C', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=[p1, p2], loc='upper right', fontsize=9)

    # --- 3. PANEL: BASINÇ (PRESSURE) ---
    ax3 = axes[2]
    ax3.plot(df['minutes'], df['pressure'], color='#6A3D9A', lw=1.2, label='Fırın İçi Basınç (Pa)')
    ax3.set_ylabel("Basınç (Pa)", fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)

    # --- 4. PANEL: AKTÜATÖRLER (FUEL & FAN) ---
    ax4 = axes[3]
    ax4_twin = ax4.twinx()
    p3, = ax4.step(df['minutes'], df['fuel'], color='#FF7F00', where='post', label='Yakıt (Fuel)')
    p4, = ax4_twin.step(df['minutes'], df['fan'], color='#444444', where='post', label='Fan Devri', alpha=0.6)
    ax4.set_ylabel("Yakıt (kg/h)", color='#FF7F00', fontweight='bold')
    ax4_twin.set_ylabel("Fan (rpm)", color='#444444', fontweight='bold')
    ax4.set_xlabel("Zaman (Dakika)", fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend(handles=[p3, p4], loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_kiln_custom_windows()
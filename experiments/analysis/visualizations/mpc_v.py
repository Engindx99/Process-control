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

    within_band = np.abs(error) <= band
    settling_index = None
    for i in range(len(within_band)):
        if all(within_band[i:]):
            settling_index = i
            break
    settling_time = time[settling_index] if settling_index is not None else np.nan
    overshoot = max(0, np.max(temp) - setpoint)

    try:
        t_low, t_high = setpoint * 0.1, setpoint * 0.9
        t1 = time[np.where(temp >= t_low)[0][0]]
        t2 = time[np.where(temp >= t_high)[0][0]]
        rise_time = t2 - t1
    except:
        rise_time = np.nan

    if settling_index is not None:
        ss_error = error[settling_index:]
        ss_mae = np.mean(np.abs(ss_error))
        ss_std = np.std(ss_error)
    else:
        ss_mae, ss_std = np.nan, np.nan

    return {
        "settling_time": settling_time, "overshoot": overshoot,
        "rise_time": rise_time, "ss_mae": ss_mae, "ss_std": ss_std
    }

def plot_kiln_custom_windows(csv_path="data/mpc_results.csv", output_dir="experiments/plot"):
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return

    # --- KLASÖR OLUŞTURMA ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Klasör oluşturuldu: {output_dir}")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if 'temp' not in df.columns and 'temperature' in df.columns:
        df['temp'] = df['temperature']
    
    df['minutes'] = df['step'] * 5 / 60 
    kpis = compute_professional_kpis(df)
    plt.style.use('seaborn-v0_8-muted')

    win_w, win_h = 600, 450

    # 1. PENCERE: SICAKLIK
    fig1 = plt.figure("1. Termal Analiz", figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(df['minutes'], df['temp'], color='#E31A1C', lw=1.5, label='Fırın Sıcaklığı (°C)')
    ax1.axhline(y=1450, color='black', linestyle='--', alpha=0.5, label='Set Point')
    ax1.set_ylim(1430, 1475)
    ax1.set_ylabel("Sıcaklık", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Dakika")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    if kpis:
        kpi_text = (f"Rise Time: {kpis['rise_time']:.1f}m | Settling: {kpis['settling_time']:.1f}m\n"
                    f"Overshoot: {kpis['overshoot']:.1f}°C | MAE: {kpis['ss_mae']:.3f}")
        ax1.set_title(f"Sıcaklık Kontrolü\n{kpi_text}")
    fig1.savefig(os.path.join(output_dir, "1_thermal_analysis.png"), dpi=300)

    # 2. PENCERE: EMİSYONLAR
    fig2 = plt.figure("2. Emisyon Analizi", figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    ax2_twin = ax2.twinx()
    p1, = ax2.plot(df['minutes'], df['o2'], color='#1F78B4', label='O2 (%)')
    p2, = ax2_twin.plot(df['minutes'], df['co2'], color='#33A02C', label='CO2 (%)', alpha=0.7)
    ax2.set_ylabel("O2 (%)", color='#1F78B4', fontweight='bold')
    ax2_twin.set_ylabel("CO2 (%)", color='#33A02C', fontweight='bold')
    ax2.set_xlabel("Dakika")
    ax2.set_title("O2 & CO2 Dengesi")
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=[p1, p2], loc='upper right')
    fig2.savefig(os.path.join(output_dir, "2_emission_analysis.png"), dpi=300)

    # 3. PENCERE: BASINÇ
    fig3 = plt.figure("3. Basınç Analizi", figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(df['minutes'], df['pressure'], color='#6A3D9A', lw=1.2, label='Fırın İçi Basınç (Pa)')
    ax3.set_ylabel("Basınç (Pa)", fontsize=10, fontweight='bold')
    ax3.set_xlabel("Dakika")
    ax3.set_title("Hava Akış Basıncı")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    fig3.savefig(os.path.join(output_dir, "3_pressure_analysis.png"), dpi=300)

    # 4. PENCERE: AKTÜATÖRLER
    fig4 = plt.figure("4. Aktüatör Analizi", figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    ax4_twin = ax4.twinx()
    p3, = ax4.step(df['minutes'], df['fuel'], color='#FF7F00', where='post', label='Yakıt (Fuel)')
    p4, = ax4_twin.step(df['minutes'], df['fan'], color='#444444', where='post', label='Fan Devri', alpha=0.6)
    ax4.set_ylabel("Yakıt (kg/h)", color='#FF7F00', fontweight='bold')
    ax4_twin.set_ylabel("Fan (rpm)", color='#444444', fontweight='bold')
    ax4.set_xlabel("Dakika")
    ax4.set_title("Yakıt & Fan Kararları")
    ax4.grid(True, alpha=0.3)
    ax4.legend(handles=[p3, p4], loc='upper right')
    fig4.savefig(os.path.join(output_dir, "4_actuator_actions.png"), dpi=300)

    print(f"Tüm grafikler '{output_dir}' klasörüne yüksek çözünürlüklü (300 DPI) olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    plot_kiln_custom_windows()
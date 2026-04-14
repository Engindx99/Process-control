import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np

def generate_report():
    # 1. Config Yükle
    try:
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error: config.yaml bulunamadı!")
        return

    csv_path = cfg['paths']['results_csv']

    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} bulunamadı! Önce main.py çalıştırılmalı.")
        return

    # 2. Veriyi Yükle
    df = pd.read_csv(csv_path)
    setpoint = cfg['mpc']['setpoint']

    # =============================================================
    # SÜTUN UYUMLULUK KONTROLÜ (SAFE HANDLING)
    # =============================================================
    # main.py içindeki isimlendirmelere göre düzeltme yapıyoruz
    if "u_rl" not in df.columns and "u_rl_residual" in df.columns:
        df["u_rl"] = df["u_rl_residual"]
    
    if "total_fuel" not in df.columns:
        df["total_fuel"] = df["u_mpc"] + df.get("u_rl", 0)

    # =============================================================
    # GRAFİKLER
    # =============================================================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    fig.suptitle(
        'Döner Fırın: Hibrit Zeka (MPC + RL) Kontrol Performansı',
        fontsize=18, fontweight='bold'
    )

    # Görsel Parametreler
    label_fs = 11
    legend_fs = 10
    grid_alpha = 0.3

    # -------------------------
    # PANEL 1 - SICAKLIK (TEMP)
    # -------------------------
    ax1.plot(df['step'], df['temp'], color='#1f77b4', linewidth=2, label='Fırın Sıcaklığı')
    ax1.axhline(setpoint, linestyle='--', color='red', alpha=0.8, label=f'Setpoint ({setpoint}°C)')
    ax1.set_ylabel("Sıcaklık (°C)", fontsize=label_fs)
    ax1.legend(loc='upper right', fontsize=legend_fs)
    ax1.grid(True, alpha=grid_alpha)
    ax1.set_title("Sıcaklık Takibi", fontsize=12, loc='left')

    # -------------------------
    # PANEL 2 - YAKIT (CONTROL INPUTS)
    # -------------------------
    # MPC ve RL katkısını üst üste gösteriyoruz (Stacking)
    ax2.plot(df['step'], df['total_fuel'], color='black', alpha=0.3, label='Toplam Yakıt')
    ax2.step(df['step'], df['u_mpc'], color='green', alpha=0.7, label='MPC Ana Kontrol')
    ax2.step(df['step'], df['u_rl'], color='orange', alpha=0.8, label='RL Düzeltme (Residual)')
    
    ax2.set_ylabel("Yakıt Akışı (m³/h)", fontsize=label_fs)
    ax2.legend(loc='upper right', fontsize=legend_fs)
    ax2.grid(True, alpha=grid_alpha)
    ax2.set_title("Kontrol Girişleri (MPC vs RL)", fontsize=12, loc='left')

    # -------------------------
    # PANEL 3 - HATA (ERROR)
    # -------------------------
    ax3.plot(df['step'], df['error'], color='purple', label='Setpoint Sapması')
    ax3.fill_between(df['step'], df['error'], color='purple', alpha=0.1)
    ax3.axhline(0, color='black', linewidth=1)
    
    ax3.set_ylabel("Hata (°C)", fontsize=label_fs)
    ax3.set_xlabel("Zaman Adımı (Step)", fontsize=label_fs)
    ax3.legend(loc='upper right', fontsize=legend_fs)
    ax3.grid(True, alpha=grid_alpha)
    ax3.set_title("Kontrol Hatası", fontsize=12, loc='left')

    # -------------------------
    # İSTATİSTİKLER (METRICS)
    # -------------------------
    mse = (df['error'] ** 2).mean()
    mae = df['error'].abs().mean()
    std_error = df['error'].std()

    stats_text = (f"Performans Özet:\n"
                  f"MSE: {mse:.2f}\n"
                  f"MAE: {mae:.2f}\n"
                  f"Std Dev: {std_error:.2f}")

    # Grafik üzerine kutucuk ekle
    plt.figtext(
        0.02, 0.02, 
        stats_text,
        fontsize=10, 
        bbox={"facecolor": "white", "edgecolor": "lightgrey", "alpha": 0.8, "pad": 8}
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # 4. Kaydet
    output_dir = "experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hybrid_performance_report.png")

    plt.savefig(output_path, dpi=300)
    print(f"✅ Rapor kaydedildi: {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_report()
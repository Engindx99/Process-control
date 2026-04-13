import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os

def generate_report():
    # 1. Read Config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    csv_path = cfg['paths']['results_csv']
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found! Run main.py first.")
        return

    # 2. Load Data
    df = pd.read_csv(csv_path)
    
    # --- 3-PANEL VERTICAL DESIGN (ENGLISH - ADJUSTED FONT SIZES) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    # Ana başlık büyük kalıyor
    fig.suptitle('Rotary Kiln: Hybrid Intelligence (MPC + RL) Control Performance', fontsize=18, fontweight='bold')

    legend_pos = (0.99, 0.98)
    # Küçültülmüş yazı boyutları (Başlıklar hariç)
    label_fs = 10    # Eksen isimleri (Temperature, Fuel vb.)
    tick_fs = 9     # Eksen rakamları (1450, 0, 1000 vb.)
    legend_fs = 9   # Kutucuk içindeki yazılar
    subtitle_fs = 12 # Alt başlıklar (Monitoring, Analysis vb.)

    # Panel 1: Temperature Tracking
    ax1.plot(df['step'], df['temp'], color='#d62728', linewidth=1.5, label='Hybrid Temperature')
    ax1.axhline(y=1450, color='black', linestyle='--', alpha=0.7, label='Target (1450°C)')
    ax1.set_ylabel("Temperature (°C)", fontsize=label_fs)
    ax1.tick_params(axis='both', labelsize=tick_fs)
    ax1.legend(loc='upper right', bbox_to_anchor=legend_pos, fontsize=legend_fs, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Fuel Consumption
    ax2.plot(df['step'], df['total_fuel'], color='#1f77b4', linewidth=1.5, label='Total Fuel (Hybrid)')
    ax2.set_ylabel("Fuel (m³/h)", fontsize=label_fs)
    ax2.tick_params(axis='both', labelsize=tick_fs)
    ax2.legend(loc='upper right', bbox_to_anchor=legend_pos, fontsize=legend_fs, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Hybrid Intelligence Error
    ax3.fill_between(df['step'], df['error'], color='orange', alpha=0.2, label='Error Area')
    ax3.plot(df['step'], df['error'], color='#ff7f0e', linewidth=1, label='Instantaneous Error (°C)')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_ylabel("Deviation (°C)", fontsize= 10)
    ax3.set_xlabel("Time Step", fontsize= 10)
    ax3.tick_params(axis='both', labelsize=tick_fs)
    ax3.legend(loc='upper right', bbox_to_anchor=legend_pos, fontsize=legend_fs, frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3)

    # Performance Metrics (Küçültülmüş metin)
    mse = (df['error']**2).mean()
    mae = df['error'].abs().mean()
    stats_text = f"MSE = {mse:.4f} | MAE = {mae:.4f}"
    plt.figtext(0.97, 0.025, stats_text, ha="right", fontsize=10, 
                bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})

    plt.subplots_adjust(hspace=0.5) 
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Adjusting spaces to prevent overlapping
    plt.subplots_adjust(hspace=0.5) 
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save & Show
    output_path = "experiments/plots/hybrid_report_en.png"
    os.makedirs("experiments/plots", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ English report generated: {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_report()
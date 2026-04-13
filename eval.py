import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os

def generate_report():

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg['paths']['results_csv']

    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found! Run main.py first.")
        return

    df = pd.read_csv(csv_path)

    # =========================
    # SAFE COLUMN HANDLING
    # =========================

    if "total_fuel" not in df.columns:
        df["total_fuel"] = df["u_mpc"] + df["u_rl"]

    if "error" not in df.columns:
        df["error"] = df["temp"] - 1450.0

    # =========================
    # PLOTS
    # =========================

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    fig.suptitle(
        'Rotary Kiln: Hybrid Intelligence (MPC + RL) Control Performance',
        fontsize=18,
        fontweight='bold'
    )

    legend_pos = (0.99, 0.98)

    label_fs = 10
    tick_fs = 9
    legend_fs = 9

    # -------------------------
    # PANEL 1 - TEMP
    # -------------------------
    ax1.plot(df['step'], df['temp'], label='Temperature')
    ax1.axhline(1450, linestyle='--', color='black', label='Setpoint')
    ax1.set_ylabel("Temperature (°C)", fontsize=label_fs)
    ax1.legend(loc='upper right', bbox_to_anchor=legend_pos, fontsize=legend_fs)
    ax1.grid(True, alpha=0.3)

    # -------------------------
    # PANEL 2 - FUEL
    # -------------------------
    ax2.plot(df['step'], df['total_fuel'], label='Total Fuel (MPC + RL)')
    ax2.set_ylabel("Fuel (m³/h)", fontsize=label_fs)
    ax2.legend(loc='upper right', bbox_to_anchor=legend_pos, fontsize=legend_fs)
    ax2.grid(True, alpha=0.3)

    # -------------------------
    # PANEL 3 - ERROR
    # -------------------------
    ax3.plot(df['step'], df['error'], label='Error')
    ax3.fill_between(df['step'], df['error'], alpha=0.2)
    ax3.axhline(0, color='black')

    ax3.set_ylabel("Deviation (°C)", fontsize=label_fs)
    ax3.set_xlabel("Time Step", fontsize=label_fs)
    ax3.legend(loc='upper right', bbox_to_anchor=legend_pos, fontsize=legend_fs)
    ax3.grid(True, alpha=0.3)

    # -------------------------
    # METRICS
    # -------------------------
    mse = (df['error'] ** 2).mean()
    mae = df['error'].abs().mean()

    stats_text = f"MSE = {mse:.4f} | MAE = {mae:.4f}"

    plt.figtext(
        0.97,
        0.02,
        stats_text,
        ha="right",
        fontsize=10,
        bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5}
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    output_path = "experiments/plots/hybrid_report_en.png"
    os.makedirs("experiments/plots", exist_ok=True)

    plt.savefig(output_path, dpi=300)
    print(f"✅ Report saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    generate_report()
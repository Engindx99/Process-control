import matplotlib.pyplot as plt
import numpy as np
import logging

from src.dt.dt import RotaryKilnDigitalTwin


# =========================================================
# LOGGER
# =========================================================
logger = logging.getLogger("DT_VISUALIZER")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =========================================================
# MAIN
# =========================================================
def run_and_plot(steps=3000):

    logger.info("Starting digital twin simulation...")

    kiln = RotaryKilnDigitalTwin(seed=42)

    steps_list = []
    temp_list = []
    o2_list = []
    fuel_list = []
    fan_list = []

    # ---------------- SIMULATION ----------------
    f_val, v_val = 16.0, 1000.0

    for i in range(steps):

        f_val = np.clip(f_val + np.random.normal(0, 0.1), 13, 21)
        v_val = np.clip(v_val + np.random.normal(0, 2.0), 960, 1080)

        record = kiln.step(f_val, v_val)

        steps_list.append(record["Step"])
        temp_list.append(record["Temperature"])
        o2_list.append(record["O2"])
        fuel_list.append(record["Fuel"])
        fan_list.append(record["Fan"])

    logger.info("Simulation completed")

    # =====================================================
    # WINDOW 1 → TEMPERATURE
    # =====================================================
    plt.figure(figsize=(14, 6))

    plt.plot(steps_list, temp_list, linewidth=1.5)
    plt.axhline(1450, linestyle="--")

    plt.title("Rotary Kiln Temperature")
    plt.xlabel("Step")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, alpha=0.3)

    # =====================================================
    # WINDOW 2 → PROCESS VARIABLES (TWIN AXIS)
    # =====================================================
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Sol eksen → O2 + Fuel
    ax1.plot(steps_list, o2_list, label="O2", linewidth=1.2)
    ax1.plot(steps_list, fuel_list, label="Fuel", linewidth=1.2)
    ax1.set_ylabel("O2 / Fuel")

    # Sağ eksen → Fan (büyük ölçek)
    ax2 = ax1.twinx()
    ax2.plot(steps_list, fan_list, label="Fan", color="red", linewidth=1.2)
    ax2.set_ylabel("Fan")

    ax1.set_title("Process Variables (Properly Scaled)")
    ax1.set_xlabel("Step")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax1.grid(True, alpha=0.3)

    # =====================================================
    # SHOW
    # =====================================================
    plt.tight_layout()
    plt.show()


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    run_and_plot(3000)
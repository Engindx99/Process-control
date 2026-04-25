import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_kiln(df):

    t = df["step"]

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # =========================
    # TEMPERATURES
    # =========================
    axs[0].plot(t, df["T1"], label="Preheat")
    axs[0].plot(t, df["T2"], label="Calcination")
    axs[0].legend()
    axs[0].set_ylabel("°C")
    axs[0].grid(True)

    axs[1].plot(t, df["T3"], color="darkred", label="Burning")
    axs[1].legend()
    axs[1].set_ylabel("°C")
    axs[1].grid(True)

    axs[2].plot(t, df["T4"], label="Cooling", color="cyan")
    axs[2].legend()
    axs[2].set_ylabel("°C")
    axs[2].grid(True)

    # =========================
    # GAS + QUALITY
    # =========================
    axs[3].plot(t, df["o2"], label="O2")
    axs[3].plot(t, df["pressure"], label="Pressure")
    axs[3].plot(t, df["fuel"], label="Fuel")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show(block=True)


# =========================
# TEST DATA (kritik eksik buydu)
# =========================
if __name__ == "__main__":

    print("Plot başlıyor...")

    df = pd.DataFrame({
        "step": np.arange(200),
        "T1": np.random.normal(800, 10, 200),
        "T2": np.random.normal(1050, 15, 200),
        "T3": np.random.normal(1450, 20, 200),
        "T4": np.random.normal(900, 10, 200),
        "o2": np.random.normal(3, 0.2, 200),
        "pressure": np.random.normal(-2, 0.1, 200),
        "fuel": np.random.normal(19, 0.5, 200),
    })

    plot_kiln(df)
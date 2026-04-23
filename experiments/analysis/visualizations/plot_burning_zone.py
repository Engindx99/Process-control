import matplotlib.pyplot as plt
from src.burning_zone.burning_zone import RotaryKilnPlant
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def plot_kiln_results(df):
    t = df["step"]

    # =========================================
    # FIGURE 1 → STATE EVOLUTION (3 grafik)
    # =========================================
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 10))
    fig1.suptitle("State Evolution", fontsize=16)

    # Temperature
    axs1[0].plot(t, df["temp"], lw=1.3)
    axs1[0].set_title("Temperature (°C)")
    axs1[0].axhline(1450, linestyle="--")
    axs1[0].grid()
    axs1[0].set_ylim(1440, 1460)

    # O2
    axs1[1].plot(t, df["o2"], lw=1.3)
    axs1[1].set_title("O2 (%)")
    axs1[1].axhline(2.2, linestyle="--")
    axs1[1].grid()

    # CO2
    axs1[2].plot(t, df["co2"], lw=1.3)
    axs1[2].set_title("CO2 (%)")
    axs1[2].grid()

    plt.tight_layout()

    # =========================================
    # FIGURE 2 → CONTROL & RELATIONS (Sadece Basınç)
    # =========================================
    
    fig2, axs2 = plt.subplots(1, 1, figsize=(12, 5)) 
    fig2.suptitle("Control & Coupling Diagnostics", fontsize=16)

    # Pressure
    axs2.plot(t, df["pressure"], lw=1.3)
    axs2.set_title("Pressure")
    axs2.axhline(-3.0, linestyle="--")
    axs2.grid()

    plt.tight_layout()
    plt.show()

plant = RotaryKilnPlant()
df = plant.run(steps=21600, fuel_cmd=18, fan_cmd=850)

plot_kiln_results(df)
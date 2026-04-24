import matplotlib.pyplot as plt
from src.burning_zone.burning_zone import RotaryKilnPlant
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def plot_kiln_results(df):
    t = df["step"]

    # =========================================
    # FIGURE 1 → STATE  (3 grafik)
    # =========================================
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 10))

    # Temperature
    axs1[0].plot(t, df["temp"], color="#D62728", lw=1.2)
    axs1[0].set_title("Temperature (°C)")
    axs1[0].axhline(1450, linestyle="--")
    axs1[0].grid()
    axs1[0].set_ylim(1430, 1470)

    # O2
    axs1[1].plot(t, df["o2"], color="#4682B4", lw=1.3)
    axs1[1].set_title("O2 (%)")
    axs1[1].axhline(2.2, linestyle="--")
    axs1[1].grid()

    # CO2
    axs1[2].plot(t, df["co2"], color="#708090", lw=1.3)
    axs1[2].set_title("CO2 (%)")
    axs1[2].grid()

    plt.tight_layout()

    # =========================================
    # FIGURE 2 → CONTROL & RELATIONS (Sadece Basınç)
    # =========================================
    
    fig2, axs2 = plt.subplots(1, 1, figsize=(12, 5)) 

    # Pressure
    axs2.plot(t, df["pressure"], lw=1.3)
    axs2.set_title("Pressure")
    axs2.axhline(-2.0, linestyle="--")
    axs2.grid()

    plt.tight_layout()
    plt.show()

plant = RotaryKilnPlant()

steps = 21600
fuel_val = 18
fan_val = 960

print(f"Simülasyon başlatılıyor: {steps} adım...")

for _ in range(steps):
    plant.step(fuel_cmd=fuel_val, fan_cmd=fan_val)

# Veriyi dijital ikizin içindeki listeden DataFrame'e dönüştürüyoruz
df = pd.DataFrame(plant.data)

# Görselleştirme fonksiyonunu çağırıyoruz
plot_kiln_results(df)
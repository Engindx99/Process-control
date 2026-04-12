import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from digital_twin.dt import RotaryKilnDigitalTwin

df = pd.read_csv("kiln_dataset.csv")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)


y_ticks_sicaklik = np.arange(1200, 1601, 100)
axes[0].plot(df["adim"], df["sicaklik"], color="red", linewidth=0.8)
axes[0].set_ylim(1200, 1600)
axes[0].set_yticks(y_ticks_sicaklik)
axes[0].set_ylabel("Sıcaklık (°C)")
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Rotary Klin Burning Zone Twin")


y_ticks_o2 = np.arange(0, 7.1, 1)
axes[1].plot(df["adim"], df["o2"], color="blue", linewidth=0.8)
axes[1].set_ylim(2,7)
axes[1].set_yticks(y_ticks_o2)
axes[1].set_ylabel("O2 (%)")
axes[1].grid(True, alpha=0.3)


y_ticks_yakit = np.arange(10, 25.1, 2)
axes[2].plot(df["adim"], df["fuel"], color="green", linewidth=0.8)
axes[2].set_ylim(10, 25)
axes[2].set_yticks(y_ticks_yakit)
axes[2].set_ylabel("Yakıt")
axes[2].set_xlabel("Adım")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("firin_ciktilari.png", dpi=150)
plt.show()
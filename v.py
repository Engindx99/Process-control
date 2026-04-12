import matplotlib.pyplot as plt
import pandas as pd
from dt import RotaryKilnDigitalTwin
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("kiln_dataset.csv")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Sıcaklık
axes[0].plot(df["adim"], df["sicaklik"], color="red", linewidth=0.8)
axes[0].set_ylim(0, 1600)
axes[0].set_ylabel("Sıcaklık (°C)")
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Rotary Klin Burning Zone Twin ")

# O2
axes[1].plot(df["adim"], df["o2"], color="blue", linewidth=0.8)
axes[1].set_ylim(0, 10)
axes[1].set_ylabel("O2 (%)")
axes[1].grid(True, alpha=0.3)

# Yakıt
axes[2].plot(df["adim"], df["fuel"], color="green", linewidth=0.8)
axes[2].set_ylim(10, 25)
axes[2].set_ylabel("Yakıt")
axes[2].set_xlabel("Adım")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("firin_ciktilari.png", dpi=150)
plt.show()
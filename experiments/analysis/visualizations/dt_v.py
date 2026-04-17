import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kiln_dataset.csv")

# ---------------- WINDOW 1: TEMPERATURE ----------------
plt.figure(figsize=(12, 6))
plt.plot(df["Step"], df["Temperature"], linewidth=1.25)

plt.title("Rotary Kiln Temperature Time Series")
plt.xlabel("Step")
plt.ylabel("Temperature (°C)")
plt.grid(True)

plt.show()

# ---------------- WINDOW 2: O2 + FUEL ----------------
plt.figure(figsize=(12, 6))

plt.plot(df["Step"], df["O2"], label="O2", linewidth=1.25)
plt.plot(df["Step"], df["Fuel"], label="Fuel", linewidth=1.25)

plt.title("O2 and Fuel Time Series")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.show()
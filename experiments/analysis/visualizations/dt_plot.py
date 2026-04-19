import matplotlib.pyplot as plt
import pandas as pd
from src.dt.dt import RotaryKilnDigitalTwin

# =========================
# DT RUN
# =========================
dt = RotaryKilnDigitalTwin()
dt.reset()

data = []

fuel_cmd = dt.fuel
fan_cmd = dt.fan

for i in range(1440):
    record = dt.step(fuel_cmd, fan_cmd)
    data.append(record)

df = pd.DataFrame(data)

# =========================
# TEMPERATURE (fixed y-axis)
# =========================
plt.figure()
plt.plot(df["step"], df["temperature"], color="red")
plt.title("Temperature")
plt.xlabel("Step")
plt.ylabel("°C")
plt.ylim(1420, 1480)   # 🔥 fixed axis
plt.grid()
plt.show()

# =========================
# O2
# =========================
plt.figure()
plt.plot(df["step"], df["o2"], color="green")
plt.title("O2")
plt.xlabel("Step")
plt.ylabel("O2 Level")
plt.grid()
plt.show()

# =========================
# EFFICIENCY
# =========================
plt.figure()
plt.plot(df["step"], df["efficiency"], color="purple")
plt.title("Efficiency")
plt.xlabel("Step")
plt.ylabel("Efficiency")
plt.grid()
plt.show()
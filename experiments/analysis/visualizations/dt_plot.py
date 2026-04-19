import matplotlib.pyplot as plt
from src.dt.dt import RotaryKilnPlant

# =========================
# RUN MULTI-EPISODE
# =========================
episodes = 1

for ep in range(episodes):

    plant = RotaryKilnPlant(seed=None)  # IMPORTANT: no seed → stochastic
    df = plant.run(steps=1440)

    t = df["step"]

    # =========================
    # O2
    # =========================
    plt.figure()
    plt.plot(t, df["o2"])
    plt.title(f"O2 Dynamics - Episode {ep+1}")
    plt.xlabel("Step")
    plt.ylabel("O2 (%)")
    plt.grid(True)
    plt.show()

# =========================
# TEMPERATURE
# =========================
    plt.figure()
    plt.plot(t, df["temp"])
    plt.title(f"Temperature - Episode {ep+1}")
    plt.xlabel("Step")
    plt.ylabel("Temp (°C)")

# 🔥 FIX: y-axis ticks 1400–1500 step 25
    plt.yticks(range(1400, 1501, 25))

    plt.grid(True)
    plt.show()

    # =========================
    # PRESSURE
    # =========================
    plt.figure()
    plt.plot(t, df["pressure"])
    plt.title(f"Draft Pressure - Episode {ep+1}")
    plt.xlabel("Step")
    plt.ylabel("Pressure")
    plt.grid(True)
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kiln_dataset.csv")

# ---------------- PENCERE 1: SICAKLIK ----------------
plt.figure(figsize=(12, 6))
plt.plot(df["adim"], df["sicaklik"], linewidth=1.5)

plt.title("Rotary Kiln Sıcaklık Zaman Serisi")
plt.xlabel("Adım")
plt.ylabel("Sıcaklık (°C)")
plt.grid(True)

plt.show()

# ---------------- PENCERE 2: O2 + YAKIT ----------------
plt.figure(figsize=(12, 6))

plt.plot(df["adim"], df["o2"], label="O2", linewidth=1.5)
plt.plot(df["adim"], df["fuel"], label="Fuel", linewidth=1.5)

plt.title("O2 ve Yakıt Zaman Serisi")
plt.xlabel("Adım")
plt.ylabel("Değer")
plt.legend()
plt.grid(True)

plt.show()
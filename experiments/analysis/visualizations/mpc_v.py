import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA LOAD
# =========================
file_path = "kiln_v10_12_results.csv"
df = pd.read_csv(file_path)

# =========================
# 1. ÖZEL PENCERE: Sıcaklık ve Yakıt
# =========================
plt.figure(figsize=(10, 8))

# Üstte Sıcaklık (1440-1460 Aralığı, 5'erli)
plt.subplot(2, 1, 1)
plt.plot(df["second"], df["temp"], color='red', label="Sıcaklık")
plt.axhline(y=1450, color='black', linestyle='--', lw=1.3, label="Setpoint (1450)")
plt.ylim(1440, 1460) 
plt.yticks(np.arange(1440, 1461, 5)) 
plt.ylabel("Temperature (°C)")
plt.title("Hassas Sıcaklık ve Yakıt Analizi (Kapatınca Diğerleri Gelecek)")
plt.legend(loc="upper right")
plt.grid(True, which='both', linestyle=':', alpha=0.6)

# Altta Yakıt
plt.subplot(2, 1, 2)
plt.plot(df["second"], df["fuel"], color='orange', label="Fuel")
plt.xlabel("Time (second)")
plt.ylabel("Fuel (kg/h)")
plt.legend(loc="upper right")
plt.grid(True)

plt.show()  # <--- İlk pencereyi gösterir ve kapanmasını bekler.

# =========================
# 2. DİĞERLERİ: Sırayla Gelen Pencereler
# =========================
other_columns = ["o2", "filt_o2", "pressure", "filt_press", "fan", "error"]

for col in other_columns:
    plt.figure()
    plt.plot(df["second"], df[col])
    plt.xlabel("Time (second)")
    plt.ylabel(col)
    plt.title(f"{col} vs Time (Kapatınca Sıradaki Gelecek)")
    plt.grid(True)
    plt.show()  # <--- Her döngüde durur, pencere kapanınca bir sonrakine geçer.
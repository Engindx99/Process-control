import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA LOAD
# =========================
file_path = "kiln_v10_12_results.csv"
df = pd.read_csv(file_path)

# =========================
# 1. KATMAN: CONTINUOUS SIGNAL (FUEL FLOW RATE) SİMÜLASYONU
# =========================
# Gerçek dünyada vana anında zıplamaz, bir zaman sabiti (tau) ile takip eder.
def simulate_flow_rate(control_signal, tau=15):
    flow_rate = np.zeros(len(control_signal))
    for i in range(1, len(control_signal)):
        # Basit bir transfer fonksiyonu ayrıklaştırması (y[n] = a*y[n-1] + (1-a)*x[n])
        alpha = 1 / (1 + tau)
        flow_rate[i] = (1 - alpha) * flow_rate[i-1] + alpha * control_signal[i]
    return flow_rate

# Eğer CSV'de hali hazırda 'flow_rate' yoksa, kontrol sinyalinden türetelim:
df['fuel_flow_rate'] = simulate_flow_rate(df['fuel'].values, tau=5)

# =========================
# 2. ÖZEL PENCERE: SICAKLIK VE YAKIT (2 KATMANLI)
# =========================
plt.figure(figsize=(12, 9))

# --- ÜST PANEL: SICAKLIK ---
plt.subplot(2, 1, 1)
plt.plot(df["second"], df["temp"], color='#D62728', lw=1.0, label="Temperature (°C)")
plt.axhline(y=1450, color='black', linestyle='--', alpha=0.6, label="Setpoint (1450)")
plt.ylim(1440, 1460)
plt.yticks(np.arange(1440, 1461, 5))
plt.ylabel("Temperature (°C)")
plt.legend(loc="upper right")
plt.grid(True, linestyle=':', alpha=0.6)

# --- ALT PANEL: YAKIT (2 KATMANLI SİNYAL) ---
plt.subplot(2, 1, 2)
# Katman 1: Kontrol Sinyali (Basamaklı/Digital Command)
plt.step(df["second"], df["fuel"], where='post', color='#FFC300', alpha=0.5, label="Fuel Command (Step)")
# Katman 2: Gerçek Akış (Sürekli/Physical Flow)
plt.plot(df["second"], df['fuel_flow_rate'], color='#E67E22', lw=2, label="Fuel Flow Rate (Continuous)")

plt.xlabel("Time (second)")
plt.ylabel("Fuel (kg/h)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA LOAD
# =========================
file_path = "kiln_v10_12_results.csv"
df = pd.read_csv(file_path)

# =========================
# 1. LAYER: SIGNAL SMOOTHING (ACTUATOR DYNAMICS)
# =========================
def simulate_flow_rate(control_signal, tau=15):
    flow_rate = np.zeros(len(control_signal))
    flow_rate[0] = control_signal[0]
    for i in range(1, len(control_signal)):
        # Discrete time first-order filter (y[n] = (1-alpha)*y[n-1] + alpha*x[n])
        alpha = 1 / (1 + tau)
        flow_rate[i] = (1 - alpha) * flow_rate[i-1] + alpha * control_signal[i]
    return flow_rate

# --- FUEL CALCULATIONS ---
df['fuel_flow_rate'] = simulate_flow_rate(df['fuel'].values, tau=40)
# Moving average for additional noise reduction
df['fuel_flow_rate'] = df['fuel_flow_rate'].rolling(window=20, center=True).mean()

# --- FAN CALCULATIONS ---
df['fan_actual_flow'] = simulate_flow_rate(df['fan'].values, tau=60)
df['fan_actual_flow'] = df['fan_actual_flow'].rolling(window=20, center=True).mean()

# --- TEMPERATURE SMOOTHING ---
# Filtering the high-frequency measurement noise
df['temp_smooth'] = df['temp'].rolling(window=15, center=True).mean()

# =========================
# 2. VISUALIZATION: TEMP, FUEL AND FAN
# =========================
fig, ax_temp = plt.subplots(2, 1, figsize=(14, 10))

# --- TOP PANEL: TEMPERATURE (Precision Analysis) ---
# Raw noisy signal in the background (low alpha)
ax_temp[0].plot(df["second"], df["temp"], color='#D62728', lw=1.0, alpha=0.2) 
# Filtered trend line
ax_temp[0].plot(df["second"], df["temp_smooth"], color='#D62728', lw=1.3, label="Kiln Temperature (Filtered)")

ax_temp[0].axhline(y=1450, color='black', linestyle='--', alpha=0.6, label="Setpoint (1450)")
ax_temp[0].set_ylim(1440, 1460)
ax_temp[0].set_yticks(np.arange(1440, 1461, 5))
ax_temp[0].set_ylabel("Temperature (°C)", color='#D62728', fontsize=12)
ax_temp[0].legend(loc="upper left")
ax_temp[0].grid(True, linestyle=':', alpha=0.6)
ax_temp[0].set_title("High-Precision Temperature and Actuator Analysis (Noise Filtered)", fontsize=14)

# --- BOTTOM PANEL: FUEL AND FAN (Dual Y-Axes) ---
ax_fuel = ax_temp[1]
ax_fan = ax_fuel.twinx() 

# -- FUEL PLOTS (Primary Y-Axis) --
ax_fuel.step(df["second"], df["fuel"], where='post', color='#FFC300', alpha=0.3, label="Fuel Command (Step)")
ax_fuel.plot(df["second"], df['fuel_flow_rate'], color='#E67E22', lw=1.3, label="Fuel Flow Rate (Continuous)")
ax_fuel.set_ylabel("Fuel (kg/h)", color='#E67E22', fontsize=12)

# -- FAN PLOTS (Secondary Y-Axis) --
ax_fan.step(df["second"], df["fan"], where='post', color='#AED6F1', alpha=0.3, label="Fan Command (Step)")
ax_fan.plot(df["second"], df['fan_actual_flow'], color='#2E86C1', lw=1.3, label="Fan Actual RPM (Continuous)")
ax_fan.set_ylabel("Fan Speed (RPM)", color='#2E86C1', fontsize=12)

# Combined Legend
lines1, labels1 = ax_fuel.get_legend_handles_labels()
lines2, labels2 = ax_fan.get_legend_handles_labels()
ax_fuel.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

ax_fuel.set_xlabel("Time (Seconds)", fontsize=11)
ax_fuel.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
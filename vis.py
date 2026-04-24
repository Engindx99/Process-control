import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA LOAD
# =========================
file_path = "kiln_v10_12_results.csv"
df = pd.read_csv(file_path)

# =========================
# 1. LAYER: SIGNAL SMOOTHING
# =========================
def simulate_flow_rate(control_signal, tau=15):
    flow_rate = np.zeros(len(control_signal))
    flow_rate[0] = control_signal[0]
    for i in range(1, len(control_signal)):
        alpha = 1 / (1 + tau)
        flow_rate[i] = (1 - alpha) * flow_rate[i-1] + alpha * control_signal[i]
    return flow_rate

df['fuel_flow_rate'] = simulate_flow_rate(df['fuel'].values, tau=40)
df['fuel_flow_rate'] = df['fuel_flow_rate'].rolling(window=20, center=True).mean()

df['fan_actual_flow'] = simulate_flow_rate(df['fan'].values, tau=60)
df['fan_actual_flow'] = df['fan_actual_flow'].rolling(window=20, center=True).mean()

df['temp_smooth'] = df['temp'].rolling(window=15, center=True).mean()

# =========================
# 2. KPI CALCULATIONS
# =========================
setpoint = 1450

df['error'] = df['temp_smooth'] - setpoint
df['abs_error'] = np.abs(df['error'])

IAE = df['abs_error'].sum()
overshoot = max(0, df['temp_smooth'].max() - setpoint)
std_temp = df['temp_smooth'].std()

# Settling
settling_band = 2
within_band = np.abs(df['error']) <= settling_band
settling_time = None
for i in range(len(within_band)):
    if within_band.iloc[i:].all():
        settling_time = df["second"].iloc[i]
        break

# =========================
# 3. KPI OPERATOR LOGIC
# =========================
limits = {
    "overshoot_good": 2,
    "overshoot_bad": 5,
    "std_good": 1.5,
    "std_bad": 3,
    "iae_good": 12000,
    "iae_bad": 25000
}

def score_linear(value, good, bad):
    if value <= good:
        return 100
    if value >= bad:
        return 0
    return 100 * (1 - (value - good) / (bad - good))

score_overshoot = score_linear(overshoot, limits["overshoot_good"], limits["overshoot_bad"])
score_std = score_linear(std_temp, limits["std_good"], limits["std_bad"])
score_iae = score_linear(IAE, limits["iae_good"], limits["iae_bad"])

total_score = (score_overshoot * 0.4 + score_std * 0.3 + score_iae * 0.3)

# Status color
if total_score > 80:
    color = "#2ECC71"
    status = "STABLE"
elif total_score > 50:
    color = "#F1C40F"
    status = "WARNING"
else:
    color = "#E74C3C"
    status = "UNSTABLE"

# Trend
valid = df[['second', 'temp_smooth']].dropna()
trend = np.polyfit(valid["second"], valid["temp_smooth"], 1)[0]
if trend > 0.0005:
    trend_symbol = "↑"
elif trend < -0.0005:
    trend_symbol = "↓"
else:
    trend_symbol = "="

# =========================
# 4. VISUALIZATION
# =========================
fig, ax_temp = plt.subplots(2, 1, figsize=(14, 10))

# --- TEMPERATURE ---
ax_temp[0].plot(df["second"], df["temp"], color='#D62728', lw=1.0, alpha=0.2)
ax_temp[0].plot(df["second"], df["temp_smooth"], color='#D62728', lw=1.3, label="Kiln Temperature (Filtered)")

ax_temp[0].axhline(y=setpoint, color='black', linestyle='--', alpha=0.6, label="Setpoint (1450)")

# Settling band
ax_temp[0].fill_between(
    df["second"],
    setpoint - settling_band,
    setpoint + settling_band,
    color='green',
    alpha=0.08,
    label="Settling Band (±2°C)"
)

# Overshoot line
ax_temp[0].axhline(
    y=setpoint + overshoot,
    color='purple',
    linestyle=':',
    alpha=0.7,
    label=f"Overshoot: {overshoot:.2f}°C"
)

ax_temp[0].set_ylim(1440, 1460)
ax_temp[0].set_ylabel("Temperature (°C)")
ax_temp[0].legend(loc="upper left")
ax_temp[0].grid(True, linestyle=':', alpha=0.6)

# --- KPI PANEL (TEK KUTU) ---
kpi_text = (
    f"STATUS: {status}\n"
    f"SCORE: {total_score:.1f}/100\n"
    f"TREND: {trend_symbol}\n\n"
    f"IAE: {IAE:.0f}\n"
    f"STD: {std_temp:.2f}\n"
    f"OVR: {overshoot:.2f}°C\n"
    f"SETTLE: {settling_time if settling_time else 'N/A'} s"
)

ax_temp[0].text(
    0.99, 0.98,
    kpi_text,
    transform=ax_temp[0].transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor=color, alpha=0.9, edgecolor='black')
)

ax_temp[0].set_title("Industrial Kiln Control Dashboard")

# --- FUEL & FAN ---
ax_fuel = ax_temp[1]
ax_fan = ax_fuel.twinx()

ax_fuel.step(df["second"], df["fuel"], where='post', color='#FFC300', alpha=0.3, label="Fuel Command")
ax_fuel.plot(df["second"], df['fuel_flow_rate'], color='#E67E22', lw=1.3, label="Fuel Flow")

ax_fan.step(df["second"], df["fan"], where='post', color='#AED6F1', alpha=0.3, label="Fan Command")
ax_fan.plot(df["second"], df['fan_actual_flow'], color='#2E86C1', lw=1.3, label="Fan RPM")

lines1, labels1 = ax_fuel.get_legend_handles_labels()
lines2, labels2 = ax_fan.get_legend_handles_labels()
ax_fuel.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

ax_fuel.set_xlabel("Time (Seconds)")
ax_fuel.set_ylabel("Fuel")
ax_fan.set_ylabel("Fan RPM")
ax_fuel.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
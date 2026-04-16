import numpy as np
import pandas as pd
import pickle


class RotaryKilnDigitalTwin:
    def __init__(self):

        self.temp = 1400.0
        self.o2 = 3.0
        self.fuel = 16.0
        self.fan = 1000.0
        self.T_env = 25.0

        self.step_count = 0
        self.data = []
        self.fuel_history = [self.fuel] * 12

        # physics
        self.thermal_mass = 100
        self.heat_gain_factor = 20.88
        self.conv_factor = 0.00022
        self.rad_factor = 5.67e-12
        self.emissivity = 0.85
        self.area_scale = 7.0

    # ---------------- O2 PHYSICS ----------------
    def calculate_o2_target(self, fuel, fan):

        fuel_effect = 2.2 * np.tanh(0.25 * (fuel - 16))
        fan_effect = 2.2 * np.tanh((fan - 1000) / 85)

        o2 = 3.0 + fan_effect - fuel_effect
        return np.clip(o2, 2.0, 5.0)

    # ---------------- COMBUSTION ----------------
    def combustion_eff(self, o2):
        return 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.2) / 1.4) ** 2)

    # ---------------- STEP ----------------
    def step(self, fuel, fan):

        self.fuel = np.clip(fuel, 12.0, 22.0)
        self.fan = np.clip(fan, 950.0, 1100.0)

        self.fuel_history.append(self.fuel)
        delayed_fuel = self.fuel_history.pop(0)

        # ---------------- O2 ----------------
        o2_target = self.calculate_o2_target(self.fuel, self.fan)
        self.o2 += 0.25 * (o2_target - self.o2)
        self.o2 = np.clip(self.o2, 2.0, 4.0)

        # ---------------- TEMPERATURE ----------------
        T_k = self.temp + 273.15
        T_env_k = self.T_env + 273.15

        comb_eff = self.combustion_eff(self.o2)

        # 🔥 düzeltme
        heat_gain = delayed_fuel * self.heat_gain_factor * comb_eff
        heat_gain = np.clip(heat_gain, 0, 2500)

        heat_loss = (0.004 + self.conv_factor * self.fan) * (self.temp - self.T_env)

        net = heat_gain - heat_loss

        # 🔥 stabil update
        self.temp += self.thermal_mass * (net / 1000.0)

        # ---------------- EFFICIENCY ----------------
        temp_eff = np.exp(-0.5 * ((self.temp - 1450) / 80) ** 2)
        o2_eff   = np.exp(-0.5 * ((self.o2 - 3.0) / 0.5) ** 2)

        eff = np.clip(0.6 * temp_eff + 0.4 * o2_eff, 0.0, 1.0)

        # ---------------- LOG ----------------
        self.data.append({
            "adim": self.step_count,
            "fuel": float(self.fuel),
            "fan": float(self.fan),
            "sicaklik": float(self.temp),
            "o2": float(self.o2),
            "efficiency": float(eff)
        })

        self.step_count += 1

        return self.temp, self.o2, eff

    # ---------------- SIM ----------------
    def run_full_simulation(self, steps=3000):

        f_val, v_val = 16.0, 1000.0

        for _ in range(steps):
            f_val = np.clip(f_val + np.random.normal(0, 0.1), 13, 21)
            v_val = np.clip(v_val + np.random.normal(0, 2.0), 960, 1080)
            self.step(f_val, v_val)

        df = pd.DataFrame(self.data)
        df.to_csv("kiln_dataset.csv", index=False)

        print("OK")
        print(f"T: {df['sicaklik'].min():.0f} - {df['sicaklik'].max():.0f}")
        print(f"O2: {df['o2'].min():.2f} - {df['o2'].max():.2f}")


if __name__ == "__main__":
    kiln = RotaryKilnDigitalTwin()
    kiln.run_full_simulation()
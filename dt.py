import numpy as np
import pandas as pd
import pickle
import os


class RotaryKilnDigitalTwin:
    def __init__(self):
        # state
        self.temp = 1400.0
        self.o2 = 4.0
        self.fuel = 16.0
        self.fan = 1000.0
        self.T_env = 25.0

        self.step_count = 0
        self.data = []
        self.fuel_history = [self.fuel] * 12

        # physics params
        self.thermal_mass = 0.10
        self.heat_gain_factor = 38.0   # 🔥 artırıldı

        self.conv_factor = 0.00025
        self.rad_factor = 5.67e-12
        self.emissivity = 0.85
        self.area_scale = 7.0

        self.phys_weight = 0.95
        self.emp_weight = 0.05

    # ---------------- O2 MODEL ----------------
    def calculate_o2(self, fuel, fan):
        base_o2 = 11.6 - 0.40 * fuel
        physical_o2 = (fan / 1000.0) * 14.0 - fuel * 0.58

        o2 = 0.7 * base_o2 + 0.3 * physical_o2
        return np.clip(o2, 1.5, 8.5)

    # ---------------- COMBUSTION ----------------
    def combustion_eff(self, o2):
        # geniş ve güvenli verim
        return 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.0) / 1.8) ** 2)

    # ---------------- STEP ----------------
    def step(self, fuel, fan):
        self.fuel = np.clip(fuel, 12.0, 22.0)
        self.fan = np.clip(fan, 950.0, 1100.0)

        # delay
        self.fuel_history.append(self.fuel)
        delayed_fuel = self.fuel_history.pop(0)

        # O2 dynamics
        target_o2 = self.calculate_o2(self.fuel, self.fan)
        self.o2 += 0.18 * (target_o2 - self.o2)

        # temperatures
        T_k = self.temp + 273.15
        T_env_k = self.T_env + 273.15

        # ---------------- COMBUSTION PHYSICS ----------------
        base_eff = self.combustion_eff(self.o2)

        # 🔥 NEW: Air-Fuel Ratio etkisi
        air_fuel_ratio = self.fan / (self.fuel + 1e-6)
        afr_eff = np.exp(-0.5 * ((air_fuel_ratio - 65) / 20) ** 2)

        comb_eff = base_eff * (0.7 + 0.3 * afr_eff)

        # 🔥 heat gain
        heat_gain = delayed_fuel * self.heat_gain_factor * comb_eff

        # base load (kiln hiçbir zaman tamamen sönmez)
        heat_gain += 80

        heat_gain = np.clip(heat_gain, 0, 1000)

        # ---------------- LOSSES ----------------
        # convection
        heat_loss_conv = (0.005 + self.conv_factor * self.fan) * (self.temp - self.T_env)

        # 🔥 Excess O2 cooling
        excess_o2 = max(0.0, self.o2 - 3.0)
        extra_cooling = excess_o2 * 0.002 * (self.temp - self.T_env)

        heat_loss_conv += extra_cooling

        # radiation (biraz yumuşatıldı)
        heat_loss_rad = (
            self.emissivity *
            self.rad_factor *
            self.area_scale *
            (T_k**4 - T_env_k**4)
        ) / 140.0

        # ---------------- ENERGY BALANCE ----------------
        net_heat = heat_gain - heat_loss_conv - heat_loss_rad

        # stabilizasyon
        net_heat = np.tanh(net_heat / 600.0) * 600.0

        # soft target (çok zayıf)
        target_temp = 1450 * np.exp(-0.5 * ((self.o2 - 3.0) / 1.2) ** 2)

        dT = self.phys_weight * net_heat + self.emp_weight * (target_temp - self.temp)

        self.temp += self.thermal_mass * dT

        # safety
        self.temp = np.clip(self.temp, 500, 1550)

        # efficiency output
        eff = np.clip(
            (1.05 - 0.00006 * self.temp) *
            np.exp(-0.5 * ((self.o2 - 3.0) / 1.2) ** 2),
            0.4, 0.98
        )

        # log
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

    # ---------------- SIMULATION ----------------
    def run_full_simulation(self, steps=5000,
                            csv_file="kiln_dataset.csv",
                            pkl_file="kiln_model.pkl"):

        print(f"Simülasyon başladı: {steps} adım")

        f_val, v_val = 16.0, 1000.0

        for _ in range(steps):
            f_val = np.clip(f_val + np.random.normal(0, 0.1), 13.0, 21.0)
            v_val = np.clip(v_val + np.random.normal(0, 2.0), 960, 1080)
            self.step(f_val, v_val)

        df = pd.DataFrame(self.data)
        df.to_csv(csv_file, index=False)

        with open(pkl_file, "wb") as f:
            pickle.dump(self, f)

        print("OK")
        print(f"T: {df['sicaklik'].min():.0f} - {df['sicaklik'].max():.0f}")
        print(f"O2: {df['o2'].min():.2f} - {df['o2'].max():.2f}")


if __name__ == "__main__":
    kiln = RotaryKilnDigitalTwin()
    kiln.run_full_simulation()
import numpy as np
import pandas as pd
import pickle
import os


class RotaryKilnDigitalTwin:
    def __init__(self):
        self.temp = 1400.0
        self.o2 = 4.0
        self.fuel = 16.0
        self.fan = 1000.0
        self.T_env = 25.0

        self.step_count = 0
        self.data = []
        self.fuel_history = [self.fuel] * 12

        self.thermal_mass = 0.002
        self.heat_gain_factor = 26

        self.conv_factor = 0.00035
        self.rad_factor = 5.67e-12
        self.emissivity = 0.85
        self.area_scale = 7.0

        self.phys_weight = 0.98
        self.emp_weight = 0.005

    def calculate_o2(self, fuel, fan):
        base_o2 = 11.6 - 0.40 * fuel
        physical_o2 = (fan / 1000.0) * 14.0 - fuel * 0.58
        o2 = 0.7 * base_o2 + 0.3 * physical_o2
        return np.clip(o2, 1.5, 8.5)

    def combustion_eff(self, o2):
        return 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.0) / 1.8) ** 2)

    def step(self, fuel, fan):

        self.fuel = np.clip(fuel, 12.0, 22.0)
        self.fan = np.clip(fan, 950.0, 1100.0)

        self.fuel_history.append(self.fuel)
        delayed_fuel = self.fuel_history.pop(0)

        target_o2 = self.calculate_o2(self.fuel, self.fan)
        self.o2 += 0.15 * (target_o2 - self.o2)

        T_k = self.temp + 273.15
        T_env_k = self.T_env + 273.15

        base_eff = self.combustion_eff(self.o2)

        air_fuel_ratio = self.fan / (self.fuel + 1e-6)
        afr_eff = np.exp(-0.5 * ((air_fuel_ratio - 65) / 20) ** 2)

        comb_eff = base_eff * (0.9 + 0.1 * afr_eff)

        heat_gain = delayed_fuel * self.heat_gain_factor * comb_eff
        bias_heat = 30 + 0.02 * (1450 - self.temp)
        heat_gain += bias_heat
        heat_gain = np.clip(heat_gain, 0, 900)

        heat_loss_conv = (0.01 + self.conv_factor * self.fan) * (self.temp - self.T_env)

        excess_o2 = max(0.0, self.o2 - 3.0)
        heat_loss_conv += excess_o2 * 0.002 * (self.temp - self.T_env)

        heat_loss_rad = (
            self.emissivity *
            self.rad_factor *
            self.area_scale *
            (T_k**4 - T_env_k**4)
        ) / 3e8

        total_loss = heat_loss_conv + heat_loss_rad

        net_heat = heat_gain - total_loss

        if self.temp > 1500:
            net_heat -= (self.temp - 1500) * 2.5

        net_heat = np.clip(net_heat, -250, 250)

        target_temp = 1450 * np.exp(-0.5 * ((self.o2 - 3.0) / 1.2) ** 2)

        dT = (
            self.phys_weight * net_heat +
            self.emp_weight * (target_temp - self.temp)
        )

        self.temp += self.thermal_mass * dT
        self.temp = np.clip(self.temp, 500, 1550)

        eff = np.clip(
            (1.05 - 0.00006 * self.temp) *
            np.exp(-0.5 * ((self.o2 - 3.0) / 1.2) ** 2),
            0.4, 0.98
        )

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

    def run_full_simulation(self, steps=3000):

        #print(f"Simülasyon başladı: {steps} adım")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        dt_dir = os.path.join(src_dir, "dt")

        os.makedirs(dt_dir, exist_ok=True)

        csv_path = os.path.join(dt_dir, "kiln_dataset.csv")
        pkl_path = os.path.join(dt_dir, "kiln_model.pkl")

        #print("📁 Kayıt yolu:", csv_path)

        f_val, v_val = 16.0, 1000.0

        for _ in range(steps):
            f_val = np.clip(f_val + np.random.normal(0, 0.1), 13.0, 21.0)
            v_val = np.clip(v_val + np.random.normal(0, 2.0), 960, 1080)
            self.step(f_val, v_val)

        df = pd.DataFrame(self.data)

        df.to_csv(csv_path, index=False)

        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

        #print("✅ Kayıt başarılı")
        #print(f"T: {df['sicaklik'].min():.0f} - {df['sicaklik'].max():.0f}")


if __name__ == "__main__":
    kiln = RotaryKilnDigitalTwin()
    kiln.run_full_simulation()
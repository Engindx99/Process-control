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
        
        self.thermal_mass = 0.10
        self.heat_gain_factor = 10.0
        self.conv_factor = 0.0003
        self.rad_factor = 5.67e-12
        
        self.phys_weight = 0.50
        self.emp_weight = 0.50

    def calculate_o2(self, fuel, fan):
        base_o2 = 11.5 - 0.395 * fuel
        physical_o2 = ((fan / 1000.0) * 15.5) - (fuel * 0.65)
        return np.clip(0.70 * base_o2 + 0.30 * physical_o2, 2.0, 8.0)

    def calculate_target_temp(self, o2, fan):
        # O2 optimum %4'te maksimum 1550°C
        o2_effect = 1550 * np.exp(-0.5 * ((o2 - 4.0) / 1.0) ** 2)
        fan_effect = (fan - 1000) / 1000 * 10
        return np.clip(o2_effect - fan_effect, 800, 1550)

    def calculate_efficiency(self, temp, o2):
        temp_eff = np.clip(1.10 - 0.0001 * temp, 0.60, 0.98)
        o2_penalty = np.exp(-0.5 * ((o2 - 4.0) / 1.5) ** 2)
        return np.clip(temp_eff * (0.80 + 0.20 * o2_penalty), 0.50, 0.98)

    def step(self, fuel, fan):
        self.fuel = np.clip(fuel, 12.0, 22.0)
        self.fan = np.clip(fan, 950.0, 1100.0)
        
        self.fuel_history.append(self.fuel)
        delayed_fuel = self.fuel_history.pop(0)
        
        target_o2 = self.calculate_o2(self.fuel, self.fan)
        self.o2 += 0.25 * (target_o2 - self.o2)
        
        T_k = self.temp + 273.15
        Te_k = self.T_env + 273.15
        
        combustion_eff = np.exp(-0.5 * ((self.o2 - 4.0) / 1.8) ** 2)
        
        heat_gain = delayed_fuel * self.heat_gain_factor * combustion_eff
        heat_loss_conv = (0.01 + self.conv_factor * self.fan) * (self.temp - self.T_env)
        heat_loss_rad = (self.rad_factor * (T_k**4 - Te_k**4)) / 100
        
        target_temp = self.calculate_target_temp(self.o2, self.fan)
        
        dT = self.phys_weight * (heat_gain - heat_loss_conv - heat_loss_rad) + \
             self.emp_weight * (target_temp - self.temp)
        
        self.temp += self.thermal_mass * dT
        self.temp = np.clip(self.temp, 400, 1600)
        eff = self.calculate_efficiency(self.temp, self.o2)
        
        self.data.append({
            "adim": self.step_count,
            "fuel": round(self.fuel, 3),
            "fan": round(self.fan, 1),
            "sicaklik": round(self.temp, 2),
            "o2": round(self.o2, 3),
            "efficiency": round(eff, 4)
        })
        self.step_count += 1
        return self.temp, self.o2, eff

    def run_full_simulation(self, steps=5000, csv_file="firin_dataset_5k.csv", pkl_file="firin_model.pkl"):
        print(f"Simülasyon başladı: {steps} adım...")
        
        f_val, v_val = 16.0, 1000.0
        
        for _ in range(steps):
            f_val = np.clip(f_val + np.random.normal(0, 0.12), 13.0, 21.0)
            v_val = np.clip(v_val + np.random.normal(0, 2.5), 960.0, 1080.0)
            self.step(f_val, v_val)
            
        df = pd.DataFrame(self.data)
        df.to_csv(csv_file, index=False)
        
        with open(pkl_file, 'wb') as f:
            pickle.dump(self, f)
            
        print(f"BAŞARILI: {len(df)} satır veri üretildi.")
        print(f"O2 aralığı: {df['o2'].min():.2f}% - {df['o2'].max():.2f}%")
        print(f"Sıcaklık aralığı: {df['sicaklik'].min():.0f}°C - {df['sicaklik'].max():.0f}°C")


if __name__ == "__main__":
    kiln_twin = RotaryKilnDigitalTwin()
    kiln_twin.run_full_simulation(steps=5000)
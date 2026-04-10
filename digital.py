import numpy as np
import pandas as pd

class RotaryKilnDigitalTwin:
    
    def __init__(self):
        self.temp = 950.0
        self.o2 = 4.5
        self.fuel = 16.0
        self.fan = 1000.0
        
        self.T_env = 25.0
        self.step_count = 0
        self.data = []
        
        self.fuel_history = [self.fuel] * 12
        
        self.k_fuel_to_o2 = -0.395
        self.b_fuel_to_o2 = 11.5
        self.k_o2_to_temp = 410.0
        self.b_o2_to_temp = -172.0
        
        self.thermal_mass = 0.07
        self.heat_gain_factor = 7.2
        self.conv_factor = 0.0004
        self.rad_factor = 1.25e-10
        
    def calculate_o2(self, fuel, fan):
        base_o2 = self.b_fuel_to_o2 + self.k_fuel_to_o2 * fuel
        
        ambient_o2_supply = (fan / 1000.0) * 15.5
        o2_consumption = fuel * 0.65
        physical_o2 = ambient_o2_supply - o2_consumption
        physical_o2 = np.clip(physical_o2, 0.1, 10.0)
        
        target_o2 = 0.70 * base_o2 + 0.30 * physical_o2
        target_o2 = np.clip(target_o2, 2.0, 8.0)
        
        return target_o2
    
    def calculate_target_temperature(self, o2, fan):
        base_temp = self.b_o2_to_temp + self.k_o2_to_temp * o2
        fan_cooling = (fan - 1000) / 1000 * 15
        target_temp = base_temp - fan_cooling
        target_temp = np.clip(target_temp, 300, 1500)
        
        return target_temp
    
    def calculate_efficiency(self, temp, o2):
        temp_efficiency = 1.25 - 0.0005 * temp
        temp_efficiency = np.clip(temp_efficiency, 0.50, 0.99)
        
        o2_penalty = np.exp(-0.5 * ((o2 - 3.5) / 2.0) ** 2)
        efficiency = temp_efficiency * (0.80 + 0.20 * o2_penalty)
        efficiency = np.clip(efficiency, 0.45, 0.99)
        
        return efficiency
    
    def step(self, fuel, fan):
        self.fuel = np.clip(fuel, 12.0, 22.0)
        self.fan = np.clip(fan, 950.0, 1100.0)
        
        self.fuel_history.append(self.fuel)
        delayed_fuel = self.fuel_history.pop(0)
        
        target_o2 = self.calculate_o2(self.fuel, self.fan)
        self.o2 += 0.25 * (target_o2 - self.o2)
        self.o2 = np.clip(self.o2, 2.0, 8.0)
        
        target_temp = self.calculate_target_temperature(self.o2, self.fan)
        
        o2_eff = np.exp(-0.5 * ((self.o2 - 3.0) / 2.0) ** 2)
        heat_gain = delayed_fuel * self.heat_gain_factor * o2_eff
        heat_loss_conv = (0.01 + self.conv_factor * self.fan) * (self.temp - self.T_env)
        heat_loss_rad = self.rad_factor * (self.temp**4 - self.T_env**4)
        dT_physical = heat_gain - heat_loss_conv - heat_loss_rad
        
        dT_combined = 0.60 * dT_physical + 0.40 * (target_temp - self.temp)
        self.temp += self.thermal_mass * dT_combined
        self.temp = np.clip(self.temp, 250, 1550)
        
        efficiency = self.calculate_efficiency(self.temp, self.o2)
        
        self.data.append({
            "adim": self.step_count,
            "fuel": round(self.fuel, 3),
            "fan": round(self.fan, 1),
            "sicaklik": round(self.temp, 2),
            "o2": round(self.o2, 3),
            "efficiency": round(efficiency, 4)
        })
        self.step_count += 1
        
        return self.temp, self.o2, efficiency
    
    def simulate(self, fuel_sequence, fan_sequence, noise_level=0.0):
        for fuel, fan in zip(fuel_sequence, fan_sequence):
            self.step(fuel, fan)
            
            if noise_level > 0:
                self.temp += np.random.normal(0, noise_level * 2)
                self.o2 += np.random.normal(0, noise_level * 0.05)
                self.temp = np.clip(self.temp, 250, 1550)
                self.o2 = np.clip(self.o2, 2.0, 8.0)
                self.data[-1]["sicaklik"] = round(self.temp, 2)
                self.data[-1]["o2"] = round(self.o2, 3)
        
        return pd.DataFrame(self.data)
    
    def reset(self):
        self.temp = 950.0
        self.o2 = 4.5
        self.fuel = 16.0
        self.fan = 1000.0
        self.step_count = 0
        self.data = []
        self.fuel_history = [self.fuel] * 12


if __name__ == "__main__":
    twin = RotaryKilnDigitalTwin()
    
    fuel_seq = np.linspace(12, 22, 500).tolist()
    fan_seq = [1000.0] * 500
    
    df = twin.simulate(fuel_seq, fan_seq)
    df.to_csv("digital_twin_output.csv", index=False)
    
    print(df[['fuel', 'sicaklik', 'o2', 'efficiency']].corr().round(3))
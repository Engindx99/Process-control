import numpy as np
from scipy.optimize import minimize

class MPC:
    def __init__(self, config=None):
        self.config = config
        
        # DT Parametreleri
        self.thermal_mass = 0.05
        self.heat_gain_factor = 20.0
        self.phys_weight = 0.95
        self.emp_weight = 0.01
        
        # MPC Ayarları
        self.horizon = 6  # Tahmin ufkunu bir tık artırdık
        self.last_fuel = 16.5
        self.last_fan = 1020.0

    def internal_predict(self, temp, o2, fuel, fan):
        T_k = temp + 273.15
        T_env_k = 25.0 + 273.15
        
        base_o2 = 11.6 - 0.40 * fuel
        physical_o2 = (fan / 1000.0) * 14.0 - fuel * 0.58
        pred_o2 = np.clip(0.7 * base_o2 + 0.3 * physical_o2, 1.5, 8.5)

        comb_eff = 0.6 + 0.4 * np.exp(-0.5 * ((pred_o2 - 3.0) / 1.8) ** 2)
        heat_gain = (fuel * self.heat_gain_factor * comb_eff) + 60.0
        
        heat_loss_conv = (0.005 + 0.00025 * fan) * (temp - 25.0)
        heat_loss_rad = (0.85 * 5.67e-12 * 7.0 * (T_k**4 - T_env_k**4)) / 5e8
        
        net_heat = np.clip(heat_gain - heat_loss_conv - heat_loss_rad, -300, 300)
        target_temp_emp = 1450 * np.exp(-0.5 * ((pred_o2 - 3.0) / 1.2) ** 2)
        
        dT = self.phys_weight * net_heat + self.emp_weight * (target_temp_emp - temp)
        return temp + self.thermal_mass * dT, pred_o2

    def cost_function(self, u, current_temp, current_o2, last_u_fuel, last_u_fan):
        temp_sim, o2_sim = current_temp, current_o2
        cost = 0
        fuel_steps = u[:self.horizon]
        fan_steps = u[self.horizon:]

        for i in range(self.horizon):
            temp_sim, o2_sim = self.internal_predict(temp_sim, o2_sim, fuel_steps[i], fan_steps[i])
            
            # 1. HATA CEZASI (Artırıldı - Hedefe odaklanma)
            cost += 5.0 * (temp_sim - 1450)**2   
            cost += 20.0 * (o2_sim - 3.0)**2     
            
            # 2. YUMUŞATMA CEZASI (Düşürüldü - Daha atik hareket için)
            if i == 0:
                cost += 2.0 * (fuel_steps[i] - last_u_fuel)**2 
                cost += 0.05 * (fan_steps[i] - last_u_fan)**2
            else:
                cost += 1.0 * (fuel_steps[i] - fuel_steps[i-1])**2 
                cost += 0.02 * (fan_steps[i] - fan_steps[i-1])**2
                
        return cost

    def optimize(self, plant):
        current_temp = plant.temp
        current_o2 = plant.o2
        self.last_fuel = plant.fuel
        self.last_fan = plant.fan
        
        initial_guess = np.concatenate([
            [self.last_fuel] * self.horizon, 
            [self.last_fan] * self.horizon
        ])
        
        # Sınırları biraz daha esnetebiliriz gerekirse
        bounds = [(12.0, 22.0)] * self.horizon + [(950.0, 1100.0)] * self.horizon
        
        res = minimize(
            self.cost_function, 
            initial_guess, 
            args=(current_temp, current_o2, self.last_fuel, self.last_fan),
            method='SLSQP', 
            bounds=bounds, 
            options={'ftol': 1e-3, 'maxiter': 5}
        )
        
        return float(res.x[0]), float(res.x[self.horizon])
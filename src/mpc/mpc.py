import numpy as np
from scipy.optimize import minimize
import copy

class MPC:
    def __init__(self, cfg):
        """
        Config sözlüğünü (cfg) alarak parametreleri yükler.
        """
        # Ufuk Ayarları
        self.prediction_horizon = cfg['mpc']['prediction_horizon']
        self.control_horizon = cfg['mpc']['control_horizon']
        self.setpoint = cfg['mpc']['setpoint']
        
        # Ağırlıklar (Weights)
        self.w_mse = cfg['mpc']['weights']['mse']
        self.w_smooth = cfg['mpc']['weights']['smoothness']
        self.w_o2 = cfg['mpc']['weights']['o2_tracking']
        self.w_energy = cfg['mpc']['weights']['energy']
        
        # Fiziksel Limitler
        self.fuel_min = 12.0
        self.fuel_max = 22.0
        self.max_delta_u = 0.15  # Manevra kabiliyetini hafif artırdık

    def internal_predict(self, current_temp, current_o2, fuel, fan):
        """
        Fırın fiziğini MPC içinde simüle eden iç model.
        """
        thermal_mass = 0.35
        heat_gain_factor = 20.0
        T_env = 25.0
        
        # O2 Tahmini
        target_o2 = np.clip((11.6 - 0.40 * fuel) * 0.7 + ((fan / 1000.0) * 14.0 - fuel * 0.58) * 0.3, 1.5, 8.5)
        pred_o2 = current_o2 + 0.18 * (target_o2 - current_o2)

        # Sıcaklık Tahmini
        T_k = current_temp + 273.15
        T_env_k = T_env + 273.15

        # Yanma Verimi
        comb_eff = (0.6 + 0.4 * np.exp(-0.5 * ((pred_o2 - 3.0) / 1.8) ** 2))
        heat_gain = fuel * heat_gain_factor * comb_eff + 60.0
        
        # Kayıplar
        h_loss_conv = (0.005 + 0.00025 * fan) * (current_temp - T_env)
        h_loss_rad = (0.85 * 5.67e-12 * 7.0 * (T_k**4 - T_env_k**4)) / 5e8

        net_heat = np.clip(heat_gain - h_loss_conv - h_loss_rad, -300, 300)
        
        # Ampirik Düzeltme
        target_temp = 1450 * np.exp(-0.5 * ((pred_o2 - 3.0) / 1.2) ** 2)
        dT = (0.95 * net_heat + 0.01 * (target_temp - current_temp))
        
        pred_temp = current_temp + thermal_mass * dT
        return pred_temp, pred_o2

    def cost_function(self, u_sequence, plant_state):
        temp_sim, o2_sim, current_fuel = plant_state
        
        u_full = np.ones(self.prediction_horizon)
        u_full[:self.control_horizon] = u_sequence
        u_full[self.control_horizon:] = u_sequence[-1]

        cost = 0
        last_u = current_fuel

        for i in range(self.prediction_horizon):
            temp_sim, o2_sim = self.internal_predict(temp_sim, o2_sim, u_full[i], 1000.0)
            
            # 1. Hedef Takip (MSE)
            # Gecikme telafisi: 12. adımdan sonra ağırlık artar
            delay_multiplier = 5.0 if i > 12 else 1.0
            cost += self.w_mse * delay_multiplier * (temp_sim - self.setpoint)**2 

            # 2. O2 Kısıtı (Tracking)
            if o2_sim < 3.0: 
                cost += self.w_o2 * (3.0 - o2_sim)**2

            # 3. Enerji Maliyeti (Energy)
            cost += self.w_energy * (u_full[i]**2)

            # 4. Yakıt Değişim Cezası (Smoothness)
            if i < self.control_horizon:
                cost += self.w_smooth * (u_full[i] - last_u)**2
                last_u = u_full[i]

        return cost

    def optimize(self, plant):
        plant_state = (plant.temp, plant.o2, plant.fuel)
        
        u0 = np.full(self.control_horizon, plant.fuel)
        bounds = [(self.fuel_min, self.fuel_max)] * self.control_horizon

        res = minimize(
            self.cost_function, 
            u0, 
            args=(plant_state,),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-4, 'disp': False}
        )

        return res.x[0] if res.success else plant.fuel
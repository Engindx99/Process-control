import numpy as np
import yaml
import logging
from scipy.optimize import minimize

class MPC:
    def __init__(self, config_path="config.yaml"):
        self.logger = logging.getLogger("MPC_Module")
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        mpc_cfg = self.cfg["mpc"]
        self.Np = mpc_cfg.get("prediction_horizon", 20)
        self.Nc = mpc_cfg.get("control_horizon", 3)
        self.set_temp = self.cfg["system"]["setpoint"]
        self.set_o2 = self.cfg["system"]["target_o2"]

        # Limitler ve Ağırlıklar
        self.fuel_min, self.fuel_max = mpc_cfg["fuel_min"], mpc_cfg["fuel_max"]
        self.fan_min, self.fan_max = mpc_cfg["fan_min"], mpc_cfg["fan_max"]
        self.max_delta_fuel = mpc_cfg["max_delta_fuel"]
        self.max_delta_fan = mpc_cfg["max_delta_fan"]
        
        self.W_temp = mpc_cfg["weight_temp"]
        self.W_o2 = mpc_cfg["weight_o2"]
        self.W_fuel_chg = mpc_cfg["weight_fuel_chg"]

        self.last_fuel = mpc_cfg["initial_fuel"]
        self.last_fan = mpc_cfg["initial_fan"]

    def internal_model_step(self, state, u, fuel_history):
        """
        DEEPCOPY YERİNE: Sadece saf matematik. 
        Plant'ın içindeki denklemleri buraya (veya ortak bir modüle) alıyoruz.
        """
        temp, o2 = state
        fuel_cmd, fan_cmd = u
        
        # Plant'taki gecikme mantığını simüle et
        eff_fuel = fuel_history[0]
        new_history = fuel_history[1:] + [fuel_cmd]

        # --- PLANT DENKLEMLERİNİN HIZLI ÖZETİ ---
        eff = 1.0 / (1.0 + np.exp(-(o2 - 2.5)))
        heat_gen = (eff_fuel * 22.0 * eff + 2000.0) / 1000
        heat_loss = (0.00037 * fan_cmd * (temp - 25)) / 1000
        new_temp = temp + 0.011 * (heat_gen - heat_loss)
        
        o2_in = 0.21 * (1.0 + 0.002 * (fan_cmd - 950))
        o2_sink = 0.055 * eff_fuel * eff
        new_o2 = o2 + 0.08 * (o2_in - o2_sink + (2.6 - o2)/6.0 + 0.02)
        
        return [new_temp, np.clip(new_o2, 0.5, 5.0)], new_history

    def objective_function(self, u_sequence, current_state, current_history):
        cost = 0.0
        state = list(current_state)
        history = list(current_history)
        u_pairs = u_sequence.reshape(-1, 2)
        u_prev = [self.last_fuel, self.last_fan]

        for i in range(self.Np):
            u_curr = u_pairs[i if i < self.Nc else self.Nc-1]
            state, history = self.internal_model_step(state, u_curr, history)
            
            cost += self.W_temp * (state[0] - self.set_temp)**2
            cost += self.W_o2 * (state[1] - self.set_o2)**2
            if i < self.Nc:
                cost += self.W_fuel_chg * (u_curr[0] - u_prev[0])**2
                u_prev = u_curr
        return cost

    def get_action(self, plant):
        """Plant nesnesinden sadece gerekli verileri alıp matematik koşturur."""
        # Nesneyi kopyalamıyoruz, sadece değerleri alıyoruz
        current_state = [plant.temp, plant.o2]
        current_history = list(plant.history_fuel) 

        u0 = np.tile([self.last_fuel, self.last_fan], self.Nc)
        bounds = [(self.fuel_min, self.fuel_max), (self.fan_min, self.fan_max)] * self.Nc
        
        res = minimize(
            self.objective_function, u0, 
            args=(current_state, current_history),
            method='SLSQP', bounds=bounds,
            options={'maxiter': 10, 'ftol': 1e-4}
        )
        
        target_u = res.x.reshape(-1, 2)[0]
        
        # Değişim limitlerini uygula
        df = np.clip(target_u[0] - self.last_fuel, -self.max_delta_fuel, self.max_delta_fuel)
        dfan = np.clip(target_u[1] - self.last_fan, -self.max_delta_fan, self.max_delta_fan)
        
        self.last_fuel = np.clip(self.last_fuel + df, self.fuel_min, self.fuel_max)
        self.last_fan = np.clip(self.last_fan + dfan, self.fan_min, self.fan_max)
        
        return self.last_fuel, self.last_fan
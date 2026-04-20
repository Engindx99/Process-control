import numpy as np
import yaml
import logging
from scipy.optimize import minimize
from typing import Union, Dict, Any

class MPC:
    def __init__(self, config_input: Union[str, Dict[str, Any]] = "config.yaml"):
        self.logger = logging.getLogger("MPC_Module")
        
        # --- Config Yükleme ---
        if isinstance(config_input, dict):
            self.cfg = config_input
        else:
            with open(config_input, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

        mpc_cfg = self.cfg.get("mpc", {})
        sys_cfg = self.cfg.get("system", {})

        # --- Kontrol Parametreleri ---
        self.Np = int(mpc_cfg.get("prediction_horizon", 80))
        self.Nc = int(mpc_cfg.get("control_horizon", 10))
        self.set_temp = float(sys_cfg.get("setpoint", 1450.0))
        self.set_o2 = float(sys_cfg.get("target_o2", 2.25))

        # --- Sınırlar ---
        self.fuel_min = float(mpc_cfg.get("fuel_min", 5.0))
        self.fuel_max = float(mpc_cfg.get("fuel_max", 45.0))
        self.fan_min = float(mpc_cfg.get("fan_min", 700.0))
        self.fan_max = float(mpc_cfg.get("fan_max", 1800.0))
        
        # --- Dinamik Ağırlıklar (Config'den veya Varsayılan) ---
        self.W_temp = float(mpc_cfg.get("weight_temp", 5184.0))
        self.W_o2 = float(mpc_cfg.get("weight_o2", 3170.0))
        self.W_fuel_chg = float(mpc_cfg.get("weight_fuel_chg", 13254.0))
        self.W_fan_chg = float(mpc_cfg.get("weight_fan_chg", 340.0))

        # Başlangıç Durumları
        self.last_fuel = float(mpc_cfg.get("initial_fuel", 14.0))
        self.last_fan = float(mpc_cfg.get("initial_fan", 950.0))

    def internal_model_step(self, state, u, fuel_history):
        """İç model: Plant ile senkronize dinamikler."""
        temp, o2 = state
        fuel_cmd, fan_cmd = u
        
        # Gecikmeli yakıt etkisini simüle et
        eff_fuel = fuel_history[0] if len(fuel_history) > 0 else fuel_cmd
        new_history = fuel_history[1:] + [fuel_cmd] if len(fuel_history) > 0 else [fuel_cmd]

        # Yanma Verimi (Sigmoid)
        eff = 1.0 / (1.0 + np.exp(-(o2 - 2.5)))
        
        # Isı ve O2 Dengesi (Plant ile tam uyumlu katsayılar)
        heat_gen = (eff_fuel * 22.0 * eff + 754.0) / 3200.0
        heat_loss = (0.0004 * fan_cmd * (temp - 25.0)) / 3200.0
        new_temp = temp + 0.004 * (heat_gen - heat_loss)
        
        o2_in = 0.21 * (1.0 + 0.002 * (fan_cmd - 950.0))
        o2_sink = 0.055 * eff_fuel * eff
        new_o2 = o2 + 0.08 * (o2_in - o2_sink + (2.6 - o2)/6.0 + 0.02)
        
        return [new_temp, np.clip(new_o2, 0.5, 5.0)], new_history

    def objective_function(self, u_sequence, current_state, current_history):
        """Normalize edilmiş maliyet fonksiyonu."""
        cost = 0.0
        state = list(current_state)
        history = list(current_history)
        u_pairs = u_sequence.reshape(-1, 2)
        u_prev = [self.last_fuel, self.last_fan]

        for i in range(self.Np):
            # Kontrol ufkundan sonrasını sabit tut (NC sonrası)
            idx = i if i < self.Nc else self.Nc-1
            u_curr = u_pairs[idx]
            
            state, history = self.internal_model_step(state, u_curr, history)
            
            # --- NORMALİZASYON ---
            # Sıcaklık farkı / 100, O2 farkı doğrudan kare
            cost += self.W_temp * ((state[0] - self.set_temp) / 100.0)**2
            cost += self.W_o2 * (state[1] - self.set_o2)**2
            
            if i < self.Nc:
                # Kontrol değişim cezaları
                cost += self.W_fuel_chg * (u_curr[0] - u_prev[0])**2
                cost += self.W_fan_chg * ((u_curr[1] - u_prev[1]) / 100.0)**2
                u_prev = u_curr
                
        return cost

    def get_action(self, current_temp, current_o2, fuel_history):
        """Optimizasyon çözücü ve aksiyon üretici."""
        current_state = [float(current_temp), float(current_o2)]
        
        # Emniyet Sınırları
        self.last_fuel = np.clip(self.last_fuel, self.fuel_min, self.fuel_max)
        self.last_fan = np.clip(self.last_fan, self.fan_min, self.fan_max)

        # Başlangıç tahmini ve Sınırlar (Nc kadar tile)
        u0 = np.tile([self.last_fuel, self.last_fan], self.Nc)
        bounds = [(self.fuel_min, self.fuel_max), (self.fan_min, self.fan_max)] * self.Nc
        
        # Çözücü Ayarları (Daha önce konuştuğumuz 20 iter / 1e-5 tol)
        res = minimize(
            self.objective_function, u0, 
            args=(current_state, list(fuel_history)),
            method='SLSQP', 
            bounds=bounds,
            options={
                'maxiter': 20, 
                'ftol': 1e-5,
                'disp': False 
            } 
        )
        
        # Çözüm analizi
        if res.success:
            target_u = res.x.reshape(-1, 2)[0]
        else:
            # Başarısızlık durumunda NaN değilse sonucu kabul et, NaN ise sabit kal
            target_u = res.x.reshape(-1, 2)[0] if not np.isnan(res.x).any() else [self.last_fuel, self.last_fan]
        
        # Dinamik Delta Limitleri (Config'den beslenir)
        max_d_fuel = float(self.cfg["mpc"].get("max_delta_fuel", 0.098))
        max_d_fan = float(self.cfg["mpc"].get("max_delta_fan", 27.0))

        # Değişim miktarını kısıtla (Delta Clipping)
        df = np.clip(target_u[0] - self.last_fuel, -max_d_fuel, max_d_fuel)
        dfan = np.clip(target_u[1] - self.last_fan, -max_d_fan, max_d_fan)
        
        # Son aksiyonu güncelle ve sınırla
        self.last_fuel = np.clip(self.last_fuel + df, self.fuel_min, self.fuel_max)
        self.last_fan = np.clip(self.last_fan + dfan, self.fan_min, self.fan_max)
        
        return float(self.last_fuel), float(self.last_fan)
import casadi as ca
import numpy as np
import yaml
from typing import Union, Dict, Any

class MPC:
    def __init__(self, config_input: Union[str, Dict[str, Any]] = "config.yaml"):
        # 1. Config Yükleme
        if isinstance(config_input, dict):
            self.cfg = config_input
        else:
            with open(config_input, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

        mpc_cfg = self.cfg.get("mpc", {})
        sys_cfg = self.cfg.get("system", {})

        # 2. Parametreler ve Ufuk Ayarları
        self.Np = int(mpc_cfg.get("prediction_horizon", 120)) 
        self.Nc = int(mpc_cfg.get("control_horizon", 20))    
        self.set_temp = float(sys_cfg.get("setpoint", 1450.0))
        self.set_o2 = float(sys_cfg.get("target_o2", 2.2))
        
        # Dijital İkiz Uyumluluk Katsayıları
        self.k_heat_plant = 0.28  
        self.C_th_plant = 3000.0  
        
        # Limitler (Optuna'dan gelen best_params ile beslenir)
        self.fuel_min, self.fuel_max = 0.0, 30.0
        self.fan_min = float(sys_cfg.get("fan_min", 0.0))
        self.fan_max = float(sys_cfg.get("fan_max", 1200.0)) 
        
        self.max_df = float(mpc_cfg.get("max_delta_fuel", 1.54))
        self.max_dfan = float(mpc_cfg.get("max_delta_fan", 111.0))

        # Ağırlıklar (Cost Function Weights)
        self.W_temp = float(mpc_cfg.get("weight_temp", 139000.0))
        self.W_o2 = float(mpc_cfg.get("weight_o2", 8800.0))
        self.W_fuel_chg = float(mpc_cfg.get("weight_fuel_chg", 17600.0))
        self.W_fan_chg = float(mpc_cfg.get("weight_fan_chg", 4400.0))

        # Başlangıç Değerleri
        self.last_fuel = float(mpc_cfg.get("initial_fuel", 18.0))
        self.last_fan = float(mpc_cfg.get("initial_fan", 900.0))

        self._setup_solver()

    def _setup_solver(self):
        self.opti = ca.Opti()
        
        # DEĞİŞKENLER (States: Temp, O2, Pressure | Controls: Fuel, Fan)
        self.U = self.opti.variable(2, self.Nc)
        self.X = self.opti.variable(3, self.Np + 1) 
        
        # PARAMETRELER
        self.P_current_state = self.opti.parameter(3)   # [temp, o2, press]
        self.P_fuel_hist = self.opti.parameter(10)     # Actuator lag history
        self.P_last_u = self.opti.parameter(2)          # [last_fuel, last_fan]

        # Başlangıç Durumu Kısıtı
        self.opti.subject_to(self.X[:, 0] == self.P_current_state)

        for k in range(self.Np):
            u_k = self.U[:, k] if k < self.Nc else self.U[:, self.Nc-1]
            temp_k = self.X[0, k]
            o2_k = self.X[1, k]
            press_k = self.X[2, k]
            
            # Gecikme (Lag) Modellemesi
            if k < 10:
                eff_fuel = self.P_fuel_hist[k]
            else:
                idx = k - 10
                eff_fuel = self.U[0, idx] if idx < self.Nc else self.U[0, self.Nc-1]

            # --- 1. Hava Akışı ve Basınç Dinamiği ---
            x_fan = (u_k[1] - 950.0) / 300.0
            air_flow = 2.0 * (1.0 / (1.0 + ca.exp(-x_fan)))
            
            resistance = 0.45 * press_k + 0.18 * (temp_k - 1400.0) / 300.0
            p_target = -3.0 + 1.8 * (air_flow - 1.0) - resistance
            next_press = press_k + (1.0 / 18.0) * (p_target - press_k)

            # --- 2. Yanma ve Isı Dengesi ---
            eff = 1.0 / (1.0 + ca.exp(-(o2_k - 2.5)))
            o2_gate = 1.0 / (1.0 + ca.exp(-11 * (o2_k - 1.0)))
            draft_effect = ca.exp(-0.0000012 * (u_k[1] - 900.0)**2)
            
            combustion = eff_fuel * o2_gate * draft_effect
            heat_gen = combustion * self.k_heat_plant
            
            # Isı Kayıpları (Radyasyon + Konveksiyon)
            T_ref = 1450.0
            rad_loss = 0.33 * ((temp_k / T_ref)**4.15 - (298.0 / T_ref)**4.15)
            conv_loss = 0.0034 * (1.0 + ca.tanh((u_k[1] - 900.0) / 350.0)) * (temp_k - 25.0)
            
            dT = (heat_gen - (rad_loss + conv_loss)) / self.C_th_plant
            next_temp = temp_k + dT
            
            # --- 3. O2 Dinamiği ---
            next_o2 = o2_k + 0.1 * (0.21 * air_flow - 0.045 * eff_fuel * eff + (2.6 - o2_k)/6.0)

            # Dinamikleri Bağla
            self.opti.subject_to(self.X[0, k+1] == next_temp)
            self.opti.subject_to(self.X[1, k+1] == next_o2)
            self.opti.subject_to(self.X[2, k+1] == next_press)

        # --- MALİYET FONKSİYONU ---
        obj = self.W_temp * ca.sumsqr(self.X[0, 1:] - self.set_temp) + \
              self.W_o2 * ca.sumsqr(self.X[1, 1:] - self.set_o2) + \
              self.W_fuel_chg * ca.sumsqr(self.U[0, 0] - self.P_last_u[0]) + \
              self.W_fan_chg * ca.sumsqr(self.U[1, 0] - self.P_last_u[1]) + \
              self.W_fuel_chg * ca.sumsqr(self.U[0, 1:] - self.U[0, :-1]) + \
              self.W_fan_chg * ca.sumsqr(self.U[1, 1:] - self.U[1, :-1])

        self.opti.minimize(obj)

        # --- KISITLAR ---
        # 1. Alan Sınırları
        self.opti.subject_to(self.opti.bounded(self.fuel_min, self.U[0, :], self.fuel_max))
        self.opti.subject_to(self.opti.bounded(self.fan_min, self.U[1, :], self.fan_max))

        # 2. Basınç Kısıtı (Endüstriyel Güvenlik)
        self.opti.subject_to(self.opti.bounded(-6.0, self.X[2, :], -1.0))

        # 3. Değişim Hızı (Delta) Kısıtları
        self.opti.subject_to(self.opti.bounded(-self.max_df, self.U[0, 0] - self.P_last_u[0], self.max_df))
        self.opti.subject_to(self.opti.bounded(-self.max_dfan, self.U[1, 0] - self.P_last_u[1], self.max_dfan))
        if self.Nc > 1:
            self.opti.subject_to(self.opti.bounded(-self.max_df, self.U[0, 1:] - self.U[0, :-1], self.max_df))
            self.opti.subject_to(self.opti.bounded(-self.max_dfan, self.U[1, 1:] - self.U[1, :-1], self.max_dfan))

        # --- SOLVER AYARLARI (IPOPT) ---
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_strategy": "adaptive"
        }
        self.opti.solver("ipopt", opts)

    def get_action(self, current_temp, current_o2, current_press, fuel_history):
        # Parametreleri Güncelle
        self.opti.set_value(self.P_current_state, ca.vertcat(current_temp, current_o2, current_press))
        self.opti.set_value(self.P_fuel_hist, ca.vertcat(*list(fuel_history)[-10:]))
        self.opti.set_value(self.P_last_u, ca.vertcat(self.last_fuel, self.last_fan))
        
        try:
            sol = self.opti.solve()
            # Warm Start için başlangıç değerlerini bir sonraki çözüme hazırla
            self.opti.set_initial(self.U, sol.value(self.U))
            self.opti.set_initial(self.X, sol.value(self.X))
            
            u_res = sol.value(self.U[:, 0])
            self.last_fuel, self.last_fan = float(u_res[0]), float(u_res[1])
        except Exception:
            # Çözülemezse son geçerli komutu koru (Safety Fallback)
            pass 

        return self.last_fuel, self.last_fan
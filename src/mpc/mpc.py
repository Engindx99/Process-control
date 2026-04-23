import casadi as ca
import numpy as np
import yaml
from typing import Union, Dict, Any

class MPC:
    def __init__(self, config_input: Union[str, Dict[str, Any]] = "config.yaml"):
        if isinstance(config_input, dict):
            self.cfg = config_input
        else:
            with open(config_input, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

        mpc_cfg = self.cfg.get("mpc", {})
        sys_cfg = self.cfg.get("system", {})

        self.Np = int(mpc_cfg.get("prediction_horizon", 100)) 
        self.Nc = int(mpc_cfg.get("control_horizon", 20))    
        self.set_temp = float(sys_cfg.get("setpoint", 1450.0))
        self.set_o2 = float(sys_cfg.get("target_o2", 2.2))
        
        # --- UYUMLULUK DÜZELTMESİ: Sabitler ---
        self.k_heat_plant = 0.28  # İkizindeki self.k_heat
        self.C_th_plant = 3000.0  # İkizindeki self.C_th
        
        # --- UYUMLULUK DÜZELTMESİ: Limitler ---
        self.fuel_min, self.fuel_max = 0.0, 25.0

        self.fan_min = float(sys_cfg.get("fan_min", 0.0))
        self.fan_max = float(sys_cfg.get("fan_max", 1200.0)) 
        
        self.max_df = float(mpc_cfg.get("max_delta_fuel", 0.15))
        self.max_dfan = float(mpc_cfg.get("max_delta_fan", 30.0))

        self.W_temp = float(mpc_cfg.get("weight_temp", 1000000.0))
        self.W_o2 = float(mpc_cfg.get("weight_o2", 300.0))
        self.W_fuel_chg = float(mpc_cfg.get("weight_fuel_chg", 75000.0))
        self.W_fan_chg = float(mpc_cfg.get("weight_fan_chg", 15000.0))

        self.last_fuel = float(mpc_cfg.get("initial_fuel", 18.0))
        self.last_fan = float(mpc_cfg.get("initial_fan", 900.0))

        self._setup_solver()

    def _setup_solver(self):
        self.opti = ca.Opti()
        
        self.U = self.opti.variable(2, self.Nc)
        self.X = self.opti.variable(2, self.Np + 1)
        
        self.P_current_state = self.opti.parameter(2)
        self.P_fuel_hist = self.opti.parameter(10) # İkizinde history_fuel 10 birim
        self.P_last_u = self.opti.parameter(2)

        self.opti.subject_to(self.X[:, 0] == self.P_current_state)

        for k in range(self.Np):
            u_k = self.U[:, k] if k < self.Nc else self.U[:, self.Nc-1]
            temp_k = self.X[0, k]
            o2_k = self.X[1, k]
            
            # Gecikme yönetimi
            if k < 10:
                eff_fuel = self.P_fuel_hist[k]
            else:
                idx = k - 10
                eff_fuel = self.U[0, idx] if idx < self.Nc else self.U[0, self.Nc-1]

            # --- UYUMLULUK DÜZELTMESİ: İkizdeki Yeni Dinamikler ---
            # 1. Yeni Air Flow Modeli (Sigmoid)
            x_fan = (u_k[1] - 950.0) / 300.0
            air_flow = 2.0 * (1.0 / (1.0 + ca.exp(-x_fan)))
            
            # 2. Yeni Yanma ve Verimlilik Modeli
            eff = 1.0 / (1.0 + ca.exp(-(o2_k - 2.5)))
            combustion = eff_fuel * (1.0 / (1.0 + ca.exp(-11 * (o2_k - 1.0)))) # o2_gate
            
            # 3. Yeni Isı Dengesi (Heat Gen - Heat Loss)
            heat_gen = combustion * self.k_heat_plant
            # Sadeleştirilmiş Isı Kaybı (MPC içinde T^4 ağır kaçabilir, doğrusal yakınsama yeterlidir)
            heat_loss = 0.004 * (temp_k - 25.0) + 0.33 * (temp_k / 1450.0) 
            
            dT = (heat_gen - heat_loss) / self.C_th_plant
            next_temp = temp_k + dT # dt = 1.0 olduğu için doğrudan ekliyoruz
            
            # 4. O2 Dinamiği
            next_o2 = o2_k + 0.1 * (0.21 * air_flow - 0.045 * eff_fuel * eff + (2.6 - o2_k)/6.0)

            self.opti.subject_to(self.X[0, k+1] == next_temp)
            self.opti.subject_to(self.X[1, k+1] == next_o2)

        # Objective ve Constraints (Aynı kaldı)
        obj = self.W_temp * ca.sumsqr(self.X[0, 1:] - self.set_temp) + \
              self.W_o2 * ca.sumsqr(self.X[1, 1:] - self.set_o2) + \
              self.W_fuel_chg * ca.sumsqr(self.U[0, 0] - self.P_last_u[0]) + \
              self.W_fan_chg * ca.sumsqr(self.U[1, 0] - self.P_last_u[1]) + \
              self.W_fuel_chg * ca.sumsqr(self.U[0, 1:] - self.U[0, :-1]) + \
              self.W_fan_chg * ca.sumsqr(self.U[1, 1:] - self.U[1, :-1])

        self.opti.minimize(obj)
        self.opti.subject_to(self.opti.bounded(self.fuel_min, self.U[0, :], self.fuel_max))
        self.opti.subject_to(self.opti.bounded(self.fan_min, self.U[1, :], self.fan_max))
        
        self.opti.subject_to(self.opti.bounded(-self.max_df, self.U[0, 0] - self.P_last_u[0], self.max_df))
        self.opti.subject_to(self.opti.bounded(-self.max_dfan, self.U[1, 0] - self.P_last_u[1], self.max_dfan))

        opts = {
            "ipopt.print_level": 0, 
            "print_time": 0, 
            "ipopt.max_iter": 80,
            "ipopt.tol": 1e-3,
            "ipopt.warm_start_init_point": "yes"
        }
        self.opti.solver("ipopt", opts)

    def get_action(self, current_temp, current_o2, fuel_history):
        # fuel_history son 10 adım olmalı
        self.opti.set_value(self.P_current_state, ca.vertcat(current_temp, current_o2))
        self.opti.set_value(self.P_fuel_hist, ca.vertcat(*list(fuel_history)[-10:]))
        self.opti.set_value(self.P_last_u, ca.vertcat(self.last_fuel, self.last_fan))
        
        try:
            sol = self.opti.solve()
            self.opti.set_initial(self.U, sol.value(self.U))
            self.opti.set_initial(self.X, sol.value(self.X))
            u_res = sol.value(self.U[:, 0])
            self.last_fuel, self.last_fan = float(u_res[0]), float(u_res[1])
        except:
            # Çözülemezse değişimi kısıtla (Safety)
            pass 

        return self.last_fuel, self.last_fan
import casadi as ca
import numpy as np

class KilnCasadiMPC:
    def __init__(self, config):
        self.cfg = config
        
        def get_val(name, default):
            return float(getattr(config, name, default))

        # Performans ve Ufuk Ayarları
        self.Np = int(get_val('prediction_horizon', 50)) 
        self.Nc = int(get_val('control_horizon', 5))
        self.pred_step = get_val('prediction_step_size', 40.0) 
        
        # Sınırlar (Optuna bunları config üzerinden besleyecek)
        self.fuel_lb = get_val('fuel_lb', 1800.0)
        self.fuel_ub = get_val('fuel_ub', 2273.15) # 2600K yerine güvenli sınır
        
        # Hedefler
        self.t_caco3 = get_val('target_caco3', 0.01)
        self.t_c2s = get_val('target_c2s', 0.20) # C2S ara hedefi eklendi
        self.t_c3s = get_val('target_c3s', 0.65) 
        
        # Ağırlıklar (Bayesian Opt. bunları optimize edecek)
        self.w_calc = get_val('weight_calc', 50000.0)
        self.w_c2s = get_val('weight_c2s', 25000.0) # Yeni ağırlık
        self.w_quality = get_val('weight_quality', 150000.0)
        self.w_smooth = get_val('weight_smooth', 2000.0) 

        self.setup_optimizer()

    def setup_optimizer(self):
        U_steps = ca.MX.sym('U_steps', self.Nc)
        P = ca.MX.sym('P', 5) # [T_s, CaCO3, CaO, C2S, C3S]
        x_sym = ca.MX.sym('x', 5)
        u_sym = ca.MX.sym('u')
        
        # --- KİNETİK MODEL (Basitleştirilmiş MPC İç Kodu) ---
        k_calc = 0.01 * ca.exp(-170000 / (8.314 * ca.fmax(x_sym[0], 600)))
        rate_calc = k_calc * x_sym[1]
        
        temp_eff = ca.fmax(0, x_sym[0] - 1450.0) 
        rate_c2s = 0.025 * temp_eff * x_sym[2] 
        rate_c3s = 0.035 * temp_eff * x_sym[3] 
        
        # --- ENERJİ VE KÜTLE DENGESİ ---
        q_transfer = 0.85 * (u_sym - x_sym[0]) 
        q_chem = (-rate_calc * 3000000) + (rate_c2s * 500000 + rate_c3s * 250000)
        
        dt_s = (q_transfer + q_chem / 1000.0) / 850.0
        
        f_next = ca.vertcat(
            dt_s, 
            -rate_calc,                                  
            (rate_calc * 0.56) - (rate_c2s * 2.1),       
            (rate_c2s * 0.75) - rate_c3s,                
            rate_c3s                                     
        )
        
        f = ca.Function('f', [x_sym, u_sym], [f_next])

        # --- MALİYET FONKSİYONU (Cost Function) ---
        obj = 0.0
        current_x = P
        for k in range(self.Np):
            u_k = U_steps[ca.fmin(k, self.Nc-1)]
            current_x = current_x + f(current_x, u_k) * self.pred_step
            
            # 1. Kalsinasyon Hatası
            obj += self.w_calc * (current_x[1] - self.t_caco3)**2
            
            # 2. C2S (Belit) Hatası - "Köprü" mineral kontrolü
            obj += self.w_c2s * (current_x[3] - self.t_c2s)**2
            
            # 3. C3S (Alit) Hatası 
            obj += self.w_quality * (current_x[4] - self.t_c3s)**2
            
            # 4. Stabilite (Düzgünlük) Cezası
            if k < self.Nc - 1:
                obj += self.w_smooth * (U_steps[k+1] - U_steps[k])**2

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 50, 
            'ipopt.tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes'
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', {'x': U_steps, 'f': obj, 'p': P}, opts)

    def compute_control(self, current_state_vector):
        p_val = ca.vertcat(*[float(v) for v in current_state_vector])
        
        # Solver'ı brülör sıcaklığı sınırlarında çalıştırıyoruz
        sol = self.solver(
            p=p_val, 
            lbx=self.fuel_lb, 
            ubx=self.fuel_ub, 
            x0=np.full(self.Nc, (self.fuel_lb + self.fuel_ub)/2)
        )
        
        return float(sol['x'][0])
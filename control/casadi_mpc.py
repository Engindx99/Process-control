import casadi as ca
import numpy as np

class KilnCasadiMPC:
    def __init__(self, config):
        self.cfg = config
        
        # Güvenli parametre erişimi
        def get_val(name, default):
            return float(getattr(config, name, default))

        self.Np = int(get_val('prediction_horizon', 100))
        self.Nc = int(get_val('control_horizon', 10))
        self.pred_step = get_val('prediction_step_size', 40.0)
        
        # YAML'dan gelen limitler (1700 - 2500K)
        self.fuel_lb = get_val('fuel_lb', 1700.0)
        self.fuel_ub = get_val('fuel_ub', 2500.0)
        
        # Hedefler ve Ağırlıklar
        self.t_caco3 = get_val('target_caco3', 0.02)
        self.t_c3s = get_val('target_c3s', 0.60)
        self.w_calc = get_val('weight_calc', 5000.0)
        self.w_quality = get_val('weight_quality', 15000.0)
        self.w_smooth = get_val('weight_smooth', 100.0)

        self.setup_optimizer()

    def setup_optimizer(self):
        # Karar Değişkeni: Kontrol ufku boyunca yakıt sıcaklıkları
        U_steps = ca.MX.sym('U_steps', self.Nc)
        # Parametre: Mevcut durum [T_s, CaCO3, CaO, C2S, C3S]
        P = ca.MX.sym('P', 5)

        x_sym = ca.MX.sym('x', 5)
        u_sym = ca.MX.sym('u')
        
        # --- KİNETİK MODEL (Fırını uyandıracak kısım) ---
        # Sıcaklık 900K (627°C) altına düştüğünde kalsinasyon durur.
        # Arrhenius terimi: k = A * exp(-E/RT)
        k_calc = 0.008 * ca.exp(-170000 / (8.314 * ca.fmax(x_sym[0], 600)))
        rate_calc = k_calc * x_sym[1]
        
        # Klinkerleşme (1250°C üstü)
        temp_eff = ca.fmax(0, x_sym[0] - 1250) / 100.0
        rate_c2s = 0.05 * temp_eff * x_sym[2]
        rate_c3s = 0.08 * temp_eff * x_sym[3]
        
        # --- ENERJİ DENGESİ ---
        # u_sym (Alev) ile x_sym[0] (Katı) arasındaki ısı transferi
        q_transfer = 0.025 * (u_sym - x_sym[0]) 
        # Endotermik kalsinasyon ısı soğurur (-)
        q_chem = (-rate_calc * 1780000) + (rate_c2s * 500000 + rate_c3s * 200000)
        
        # Sıcaklık değişimi (Basitleştirilmiş termal kütle)
        dt_s = (q_transfer + q_chem / 1500.0) / 1000.0 
        
        f = ca.Function('f', [x_sym, u_sym], 
                        [ca.vertcat(dt_s, 
                                   -rate_calc, 
                                   (rate_calc * 0.56) - rate_c2s - rate_c3s, 
                                   rate_c2s - rate_c3s, 
                                   rate_c3s)])

        # --- MALİYET FONKSİYONU ---
        obj = 0.0
        current_x = P
        for k in range(self.Np):
            u_k = U_steps[ca.fmin(k, self.Nc-1)]
            current_x = current_x + f(current_x, u_k) * self.pred_step
            
            # Hedeflere yakınlığı cezalandır
            obj += self.w_calc * (current_x[1] - self.t_caco3)**2
            obj += self.w_quality * (current_x[4] - self.t_c3s)**2
            
            # Yumuşak geçiş cezası
            if k < self.Nc - 1:
                obj += self.w_smooth * (U_steps[k+1] - U_steps[k])**2

        self.solver = ca.nlpsol('solver', 'ipopt', {'x': U_steps, 'f': obj, 'p': P},
                                {'ipopt.print_level': 0, 'print_time': 0})

    def compute_control(self, current_state_vector):
        p_val = ca.vertcat(*[float(v) for v in current_state_vector])
        
        # Çözücüye sınırları ve başlangıç tahminini veriyoruz
        sol = self.solver(p=p_val, 
                          lbx=self.fuel_lb, 
                          ubx=self.fuel_ub, 
                          x0=np.full(self.Nc, (self.fuel_lb + self.fuel_ub)/2))
        
        u_opt = float(sol['x'][0])
        return max(min(u_opt, self.fuel_ub), self.fuel_lb)
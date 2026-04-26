import numpy as np
from core.physics import calc_convection, calc_radiation
from core.kinetics import calc_calcination_rate, calc_reaction_heat, calc_clinker_kinetics

class KilnSolver:
    def __init__(self, config):
        self.cfg = config
        self.dx = config.length / config.n_zones

    def solve_step(self, state):
        """
        Explicit Euler yöntemi ile zaman adımı ilerletir.
        Kalsinasyon ve Sinterleme (C2S, C3S) reaksiyonlarını içerir.
        """
        # Mevcut state kopyaları
        new_T_s = state.T_solid.copy()
        new_T_g = state.T_gas.copy()
        new_c_caco3 = state.c_caco3.copy()
        new_c_cao = state.c_cao.copy()
        new_c_c2s = state.c_c2s.copy()
        new_c_c3s = state.c_c3s.copy()
        
        dt = self.cfg.dt
        v_s = self.cfg.velocity
        v_g = -5.0  # Karşı akıntı gaz hızı

        for i in range(self.cfg.n_zones):
            # --- 1. Fiziksel ve Kimyasal Hesaplamalar ---
            q_conv = calc_convection(state.T_gas[i], state.T_solid[i])
            q_rad = calc_radiation(state.T_gas[i], state.T_solid[i])
            
            # A. Kalsinasyon (Endotermik)
            rate_calc = calc_calcination_rate(state.T_solid[i], state.c_caco3[i])
            q_calc = calc_reaction_heat(rate_calc)
            
            # B. Sinterleme (Ekzotermik) - CaO, C2S ve C3S oluşumu
            r_c2s, r_c3s, q_sinter = calc_clinker_kinetics(
                state.T_solid[i], state.c_cao[i], state.c_c2s[i], state.c_c3s[i]
            )

            # --- 2. Katı Fazı Güncelleme (Enerji ve Kütle) ---
            if i > 0:
                # Enerji Dengesi (Taşınım + Işınım - Kalsinasyon_Isısı + Sinterleme_Isısı)
                advection_s = v_s * (state.T_solid[i] - state.T_solid[i-1]) / self.dx
                
                # Net ısı akısı (q_sinter pozitif yani sisteme ısı veriyor)
                dT_s = (q_conv + q_rad - q_calc + q_sinter) / (self.cfg.rho_solid * self.cfg.cp_solid) - advection_s
                new_T_s[i] += dT_s * dt
                
                # Kütle Dengesi (Adveksiyon + Reaksiyon)
                def get_adv(val_array):
                    return v_s * (val_array[i] - val_array[i-1]) / self.dx

                # CaCO3 Tüketimi
                new_c_caco3[i] += (-rate_calc - get_adv(state.c_caco3)) * dt
                
                # CaO Dengesi (Kalsinasyondan gelir [0.56 mol oranı], Sinterlemede harcanır)
                new_c_cao[i] += (rate_calc * 0.56 - r_c2s - r_c3s - get_adv(state.c_cao)) * dt
                
                # C2S Dengesi
                new_c_c2s[i] += (r_c2s - r_c3s - get_adv(state.c_c2s)) * dt
                
                # C3S Dengesi
                new_c_c3s[i] += (r_c3s - get_adv(state.c_c3s)) * dt

            # --- 3. Gaz Fazı Güncelleme ---
            if i < self.cfg.n_zones - 1:
                advection_g = v_g * (state.T_gas[i+1] - state.T_gas[i]) / self.dx
                dT_g = (-q_conv - q_rad) / (1.2 * self.cfg.cp_gas) - advection_g
                new_T_g[i] += dT_g * dt

        # Negatif konsantrasyonları ve taşmaları engelle (Sayısal stabilite için)
        state.c_caco3 = np.clip(new_c_caco3, 0, 1)
        state.c_cao = np.clip(new_c_cao, 0, 1)
        state.c_c2s = np.clip(new_c_c2s, 0, 1)
        state.c_c3s = np.clip(new_c_c3s, 0, 1)
        
        return new_T_s, new_T_g, state.c_caco3 # main.py uyumluluğu için return formatı
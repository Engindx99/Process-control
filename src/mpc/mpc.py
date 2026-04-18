import numpy as np
from scipy.optimize import minimize
from numba import njit
import logging

# =========================================================
# FAST MODEL (NUMBA) - Termal Kütle 170'e Göre Güncellendi
# =========================================================
@njit
def fast_internal_predict(temp, o2, fuel, fan):
    # Yakıt ve Fan Etkileri
    fuel_effect = 2.2 * np.tanh(0.25 * (fuel - 16))
    fan_effect = 2.2 * np.tanh((fan - 1000) / 85)

    o2_target = 3.0 + fan_effect - fuel_effect
    
    # O2 Sınırlandırma
    o2_target = max(2.0, min(5.0, o2_target))
    o2 = o2 + 0.25 * (o2_target - o2)
    o2 = max(2.0, min(4.0, o2))

    # Yanma Verimi (Combustion Efficiency)
    comb_eff = 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.2) / 1.4) ** 2)

    # Isı Kazancı ve Kaybı
    heat_gain = min(fuel * 20.88 * comb_eff, 2500.0)
    heat_loss = (0.004 + 0.00022 * fan) * (temp - 25.0)

    # KRİTİK GÜNCELLEME: 1/3100 (Thermal Mass) ~= 0.000322
    temp_next = temp + 0.000322 * (heat_gain - heat_loss)

    return temp_next, o2

# =========================================================
# COST FUNCTION - Receding Horizon & Zero-Order Hold
# =========================================================
@njit
def fast_cost(
    u, p_horizon, temp0, o20, temp_target,
    last_fuel, last_fan, w_mse, w_o2, w_energy, w_smooth
):
    # u dizisi control_horizon (c_horizon) uzunluğundadır
    c_horizon = len(u) // 2 
    fuel_seq = u[:c_horizon]
    fan_seq = u[c_horizon:]

    temp = temp0
    o2 = o20
    cost = 0.0

    for k in range(p_horizon):
        # Tahmin ufkunda kontrol ufkunu geçtiysek, son kararı sabit tut (ZOH)
        if k < c_horizon:
            curr_fuel = fuel_seq[k]
            curr_fan = fan_seq[k]
        else:
            curr_fuel = fuel_seq[c_horizon - 1]
            curr_fan = fan_seq[c_horizon - 1]

        temp, o2 = fast_internal_predict(temp, o2, curr_fuel, curr_fan)

        # 1. Hata maliyeti (Hassas hedef takibi)
        cost += w_mse * (temp - temp_target) ** 2
        
        # 2. O2 takibi
        cost += w_o2 * (o2 - 3.2) ** 2
        
        # 3. Enerji maliyeti (Aşırı yakıt kullanımını önler)
        cost += w_energy * (curr_fuel - 16.0) ** 2

        # 4. Değişim Yumuşatma (Sadece kontrol kararları için ceza)
        if k < c_horizon:
            if k == 0:
                diff_f = curr_fuel - last_fuel
                diff_v = curr_fan - last_fan
            else:
                diff_f = curr_fuel - fuel_seq[k-1]
                diff_v = curr_fan - fan_seq[k-1]
            
            # Smoothness ağırlığı burada devreye girer
            cost += w_smooth * (diff_f**2 + 0.05 * diff_v**2)

    return cost

# =========================================================
# MPC CLASS
# =========================================================
class MPC:
    def __init__(self, config):
        self.config = config["mpc"]
        self.temp_target = float(config["system"]["setpoint"])
        
        # Horizon Ayarları
        self.p_horizon = int(self.config["prediction_horizon"])
        self.c_horizon = int(self.config.get("control_horizon", 6))
        
        # Optimizasyon Parametreleri
        self.max_iter = int(self.config.get("maxiter", 15))
        
        w = self.config["weights"]
        self.w_mse = float(w["mse"])
        self.w_smooth = float(w["smoothness"])
        self.w_o2 = float(w["o2_tracking"])
        self.w_energy = float(w["energy"])

    def optimize(self, plant):
        temp0 = plant.temp
        o20 = plant.o2
        
        # Başlangıç tahmini (Warm start)
        x0 = np.concatenate([
            np.full(self.c_horizon, plant.fuel),
            np.full(self.c_horizon, plant.fan)
        ])

        # Fiziksel Sınırlar (Bounds)
        bounds = (
            [(12.0, 22.0)] * self.c_horizon +   # Yakıt min/max
            [(950.0, 1100.0)] * self.c_horizon  # Fan min/max
        )

        # Optimizasyon (SLSQP)
        res = minimize(
            fast_cost,
            x0,
            args=(
                self.p_horizon, temp0, o20, self.temp_target,
                plant.fuel, plant.fan,
                self.w_mse, self.w_o2, self.w_energy, self.w_smooth
            ),
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": self.max_iter,
                "ftol": 1e-4
            }
        )

        # İlk adımı (current action) al ve uygula
        u = res.x
        fuel_cmd = float(u[0])
        fan_cmd = float(u[self.c_horizon])

        return fuel_cmd, fan_cmd
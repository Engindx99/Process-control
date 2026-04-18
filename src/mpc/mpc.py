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
    # MPC artık fırının gerçek hızını biliyor.
    temp_next = temp + 0.000322 * (heat_gain - heat_loss)

    return temp_next, o2

# =========================================================
# COST FUNCTION - Kararlı Durum Hatasını Silmek İçin Optimize Edildi
# =========================================================
@njit
def fast_cost(
    u, horizon, temp0, o20, temp_target,
    last_fuel, last_fan, w_mse, w_o2, w_energy, w_smooth
):
    fuel_seq = u[:horizon]
    fan_seq = u[horizon:]

    temp = temp0
    o2 = o20
    cost = 0.0

    for k in range(horizon):
        temp, o2 = fast_internal_predict(temp, o2, fuel_seq[k], fan_seq[k])

        # Hata maliyeti (Karesel hata 1450'ye zorlar)
        cost += w_mse * (temp - temp_target) ** 2
        
        # O2 takibi
        cost += w_o2 * (o2 - 3.2) ** 2
        
        # Enerji maliyeti (Aşırı yakıt kullanımını önler ama MSE'yi ezmemeli)
        cost += w_energy * (fuel_seq[k] - 16.0) ** 2

        # Değişim Yumuşatma (Smoothness)
        if k == 0:
            diff_f = fuel_seq[k] - last_fuel
            diff_v = fan_seq[k] - last_fan
        else:
            diff_f = fuel_seq[k] - fuel_seq[k-1]
            diff_v = fan_seq[k] - fan_seq[k-1]
            
        cost += w_smooth * (diff_f**2 + 0.05 * diff_v**2)

    return cost

# =========================================================
# MPC CLASS
# =========================================================
class MPC:
    def __init__(self, config):
        self.config = config["mpc"]
        self.temp_target = float(config["system"]["setpoint"])
        self.horizon = int(self.config["prediction_horizon"])

        w = self.config["weights"]
        self.w_mse = float(w["mse"])
        self.w_smooth = float(w["smoothness"])
        self.w_o2 = float(w["o2_tracking"])
        self.w_energy = float(w["energy"])

        self.step_count = 0

    def optimize(self, plant):
        temp0 = plant.temp
        o20 = plant.o2
        
        # Başlangıç tahmini (Warm start için mevcut değerler)
        x0 = np.concatenate([
            np.full(self.horizon, plant.fuel),
            np.full(self.horizon, plant.fan)
        ])

        # Fiziksel Sınırlar (Bounds)
        bounds = (
            [(12.0, 22.0)] * self.horizon +   # Yakıt sınırları
            [(950.0, 1100.0)] * self.horizon  # Fan devri sınırları
        )

        # Optimizasyon (SLSQP)
        res = minimize(
            fast_cost,
            x0,
            args=(
                self.horizon, temp0, o20, self.temp_target,
                plant.fuel, plant.fan,
                self.w_mse, self.w_o2, self.w_energy, self.w_smooth
            ),
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": 5, 
                "ftol": 1e-4    # Tolerans sıkılaştırıldı
            }
        )

        u = res.x
        fuel_cmd = float(u[0])
        fan_cmd = float(u[self.horizon])

        self.step_count += 1
        return fuel_cmd, fan_cmd
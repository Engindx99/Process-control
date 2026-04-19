import numpy as np
from scipy.optimize import minimize
from numba import njit

# =========================================================
# 1. FAST INTERNAL MODEL (NUMBA - CACHED)
# =========================================================
@njit(cache=True)
def fast_internal_predict(temp, o2, fuel, fan):
    """
    MPC'nin geleceği tahmin etmek için kullandığı hızlı model.
    cache=True ile tekrar derleme (re-compilation) süresinden tasarruf sağlar.
    """
    # Yakıt ve Fan Etkileri
    fuel_effect = 1.2 * np.tanh(0.18 * (fuel - 16))
    fan_effect = 1.5 * np.tanh((fan - 1000) / 120)

    # O2 Dinamiği
    o2_target = 3.4 + fan_effect - fuel_effect
    o2_target = max(2.0, min(5.0, o2_target))
    o2 = o2 + 0.12 * (o2_target - o2)

    # Yanma Verimi
    comb_eff = 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.2) / 1.4) ** 2)

    # Enerji Dengesi
    heat_gain = fuel * 22.85 * comb_eff
    heat_loss = (0.004 + 0.00030 * fan) * (temp - 25.0)

    # Termal Kütle Etkisi (1 / 1050)
    temp_next = temp + 0.00095238 * (heat_gain - heat_loss)

    return temp_next, o2

# =========================================================
# 2. COST FUNCTION (NUMBA - CACHED)
# =========================================================
@njit(cache=True)
def fast_cost(
    u, p_horizon, temp0, o20, temp_target,
    last_fuel, last_fan, w_mse, w_o2, w_energy, w_smooth
):
    c_horizon = len(u) // 2 
    fuel_seq = u[:c_horizon]
    fan_seq = u[c_horizon:]

    temp = temp0
    o2 = o20
    cost = 0.0

    for k in range(p_horizon):
        # Zero-Order Hold (ZOH)
        idx = min(k, c_horizon - 1)
        curr_fuel = fuel_seq[idx]
        curr_fan = fan_seq[idx]

        temp, o2 = fast_internal_predict(temp, o2, curr_fuel, curr_fan)

        # Hata maliyeti
        cost += w_mse * (temp - temp_target) ** 2
        
        # O2 Takibi
        cost += w_o2 * (o2 - 3.2) ** 2
        
        # Enerji (Yakıt Tasarrufu)
        cost += w_energy * (curr_fuel - 16.0) ** 2

        # Değişim Yumuşatma
        if k < c_horizon:
            prev_f = last_fuel if k == 0 else fuel_seq[k-1]
            prev_v = last_fan if k == 0 else fan_seq[k-1]
            cost += w_smooth * ((curr_fuel - prev_f)**2 + 0.1 * (curr_fan - prev_v)**2)

    return cost

# =========================================================
# 3. MPC CLASS (HIZLI VE DINAMIK)
# =========================================================
class MPC:
    def __init__(self, config):
        self.full_config = config
        self.mpc_config = config["mpc"]
        
        # Ufuk Ayarları (Hız için ideal değerler)
        self.p_horizon = int(self.mpc_config.get("prediction_horizon", 120))
        self.c_horizon = int(self.mpc_config.get("control_horizon", 10)) # 20'den 10'a düşürmek hızı 2 kat artırır
        self.max_iter = int(self.mpc_config.get("maxiter", 12))         # 20'den 12'ye düşürmek optimizasyonu hızlandırır
        
        # Ağırlıklar
        w = self.mpc_config["weights"]
        self.w_mse = float(w["mse"])
        self.w_smooth = float(w["smoothness"])
        self.w_o2 = float(w["o2_tracking"])
        self.w_energy = float(w.get("energy", 0.5))

    def optimize(self, plant, current_target):
        """
        Dinamik limitlerle hızlı optimizasyon.
        """
        temp0 = plant.temp
        o20 = plant.o2
        
        # Limitleri config'den çek (28.0 yakıt tavanı dahil)
        limits = self.full_config.get("limits", self.full_config.get("plant", {}))
        f_min = limits.get("fuel_min", 12.0)
        f_max = limits.get("fuel_max", 28.0)
        v_min = limits.get("fan_min", 850.0)
        v_max = limits.get("fan_max", 1100.0)

        # Başlangıç tahmini (Warm start)
        x0 = np.concatenate([
            np.full(self.c_horizon, plant.fuel),
            np.full(self.c_horizon, plant.fan)
        ])

        # Sınırlar
        bounds = (
            [(f_min, f_max)] * self.c_horizon +   
            [(v_min, v_max)] * self.c_horizon
        )

        # Optimizasyon Çözücü
        res = minimize(
            fast_cost,
            x0,
            args=(
                self.p_horizon, temp0, o20, current_target,
                plant.fuel, plant.fan,
                self.w_mse, self.w_o2, self.w_energy, self.w_smooth
            ),
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": self.max_iter, 
                "ftol": 1e-3,        # Çok küçük değerler (1e-6) hızı düşürür, 1e-3 yeterlidir
                "disp": False
            }
        )

        u = res.x
        
        # Güvenlik Kırpması (Hard Clip)
        final_fuel = np.clip(u[0], f_min, f_max)
        final_fan = np.clip(u[self.c_horizon], v_min, v_max)

        return float(final_fuel), float(final_fan)
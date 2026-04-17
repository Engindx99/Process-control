import numpy as np
from scipy.optimize import minimize
from numba import njit
import logging


# =========================================================
# FAST MODEL (NUMBA)
# =========================================================
@njit
def fast_internal_predict(temp, o2, fuel, fan):

    fuel_effect = 2.2 * np.tanh(0.25 * (fuel - 16))
    fan_effect = 2.2 * np.tanh((fan - 1000) / 85)

    o2_target = 3.0 + fan_effect - fuel_effect

    if o2_target < 2.0:
        o2_target = 2.0
    if o2_target > 5.0:
        o2_target = 5.0

    o2 = o2 + 0.25 * (o2_target - o2)

    if o2 < 2.0:
        o2 = 2.0
    if o2 > 4.0:
        o2 = 4.0

    comb_eff = 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.2) / 1.4) ** 2)

    heat_gain = fuel * 20.88 * comb_eff
    if heat_gain > 2500:
        heat_gain = 2500

    heat_loss = (0.004 + 0.00022 * fan) * (temp - 25.0)

    temp_next = temp + 0.02 * (heat_gain - heat_loss)

    return temp_next, o2


# =========================================================
# COST FUNCTION
# =========================================================
@njit
def fast_cost(
    u, horizon,
    temp0, o20,
    temp_target,
    last_fuel, last_fan,
    w_mse, w_o2, w_energy, w_smooth
):

    fuel_seq = u[:horizon]
    fan_seq = u[horizon:]

    temp = temp0
    o2 = o20
    cost = 0.0

    for k in range(horizon):

        temp, o2 = fast_internal_predict(temp, o2, fuel_seq[k], fan_seq[k])

        cost += w_mse * (temp - temp_target) ** 2
        cost += w_o2 * (o2 - 3.0) ** 2
        cost += w_energy * fuel_seq[k] ** 2

        if k == 0:
            cost += w_smooth * (
                (fuel_seq[k] - last_fuel) ** 2 +
                0.1 * (fan_seq[k] - last_fan) ** 2
            )
        else:
            cost += w_smooth * (
                (fuel_seq[k] - fuel_seq[k - 1]) ** 2 +
                0.05 * (fan_seq[k] - fan_seq[k - 1]) ** 2
            )

    return cost


# =========================================================
# MPC CLASS
# =========================================================
class MPC:

    def __init__(self, config):

        self.config = config["mpc"]

        # ---------------- SETPOINT SAFE LOAD ----------------
        self.temp_target = self._get_setpoint(config)

        self.horizon = int(self.config["prediction_horizon"])

        w = self.config["weights"]
        self.w_mse = float(w["mse"])
        self.w_smooth = float(w["smoothness"])
        self.w_o2 = float(w["o2_tracking"])
        self.w_energy = float(w["energy"])

        self.last_fuel = 16.0
        self.last_fan = 1000.0
        self.step_count = 0

    # =====================================================
    # CONFIG SAFE ACCESS
    # =====================================================
    def _get_setpoint(self, config):

        # primary source
        if "system" in config and "setpoint" in config["system"]:
            return float(config["system"]["setpoint"])

        # fallback (legacy support)
        if "mpc" in config and "setpoint" in config["mpc"]:
            return float(config["mpc"]["setpoint"])

        raise KeyError("setpoint not found in config (system.setpoint required)")

    # =====================================================
    # OPTIMIZATION
    # =====================================================
    def optimize(self, plant):

        temp0 = plant.temp
        o20 = plant.o2

        self.last_fuel = plant.fuel
        self.last_fan = plant.fan

        x0 = np.concatenate([
            np.full(self.horizon, self.last_fuel),
            np.full(self.horizon, self.last_fan)
        ])

        bounds = (
            [(12.0, 22.0)] * self.horizon +
            [(950.0, 1100.0)] * self.horizon
        )

        res = minimize(
            fast_cost,
            x0,
            args=(
                self.horizon,
                temp0,
                o20,
                self.temp_target,
                self.last_fuel,
                self.last_fan,
                self.w_mse,
                self.w_o2,
                self.w_energy,
                self.w_smooth
            ),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 5, "ftol": 1e-3}
        )

        u = res.x

        fuel_cmd = float(u[0])
        fan_cmd = float(u[self.horizon])

        if self.step_count % 100 == 0:
            error = temp0 - self.temp_target
            print(f"[MPC] step={self.step_count} temp={temp0:.2f} error={error:.2f}")

        self.step_count += 1

        return fuel_cmd, fan_cmd
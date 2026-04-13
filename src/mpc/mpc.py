import numpy as np
import json
from scipy.optimize import minimize
from src.digital_twin.dt import RotaryKilnDigitalTwin


class MPC:
    def __init__(self, prediction_horizon=30, control_horizon=3):
        self.P = prediction_horizon
        self.M = control_horizon

        self.setpoint = 1450

        # cost weights
        self.lambda_u = 120.0

        # logging
        self.log = []

    # =========================
    # COST FUNCTION
    # =========================
    def _objective(self, fuel_sequence, plant):

        sim = self._clone(plant)

        temps = []
        o2_values = []

        # RL disturbance awareness (soft coupling)
        rl_bias = 0.0  # residual RL etkisi varsayımı (0-centered)

        full_sequence = np.zeros(self.P)
        full_sequence[:self.M] = fuel_sequence
        full_sequence[self.M:] = fuel_sequence[-1]

        for i in range(self.P):

            temp, o2, _ = sim.step(full_sequence[i] + rl_bias, 1000.0)

            temps.append(temp)
            o2_values.append(o2)

        temps = np.array(temps)
        o2_values = np.array(o2_values)

        # =========================
        # 1. TRACKING ERROR
        # =========================
        mse = np.mean((temps - self.setpoint) ** 2)

        # =========================
        # 2. O2 COUPLING PENALTY
        # =========================
        o2_penalty = 5.0 * np.mean((o2_values - 3.0) ** 2)

        # =========================
        # 3. SMOOTHNESS
        # =========================
        diffs = np.diff(np.concatenate([[plant.fuel], fuel_sequence]))
        smoothness = self.lambda_u * np.sum(diffs ** 2)

        # =========================
        # 4. ENERGY PENALTY (optional realism)
        # =========================
        energy_penalty = 0.01 * np.sum(fuel_sequence ** 2)

        return mse + smoothness + o2_penalty + energy_penalty

    # =========================
    # OPTIMIZATION
    # =========================
    def optimize(self, plant):

        initial_guess = np.full(self.M, plant.fuel)

        bounds = [(12.0, 22.0) for _ in range(self.M)]

        res = minimize(
            self._objective,
            initial_guess,
            args=(plant,),
            bounds=bounds,
            method='SLSQP'
        )

        target_fuel = res.x[0]

        # =========================
        # ADAPTIVE SLEW RATE
        # =========================
        current_fuel = plant.fuel

        temp_error = abs(plant.temp - self.setpoint)

        max_step = np.clip(temp_error / 1000.0, 0.02, 0.1)

        diff = target_fuel - current_fuel

        if abs(diff) > max_step:
            actual_fuel = current_fuel + np.sign(diff) * max_step
        else:
            actual_fuel = target_fuel

        # =========================
        # DEADZONE
        # =========================
        if abs(actual_fuel - current_fuel) < 0.01:
            return current_fuel

        return actual_fuel

    # =========================
    # CLONE (DIGITAL TWIN COPY)
    # =========================
    def _clone(self, plant):

        sim = RotaryKilnDigitalTwin()

        sim.temp = plant.temp
        sim.o2 = plant.o2
        sim.fuel = plant.fuel
        sim.fan = plant.fan

        if hasattr(plant, "fuel_history"):
            sim.fuel_history = plant.fuel_history.copy()

        return sim

    # =========================
    # LOGGING
    # =========================
    def log_step(self, step, fuel, temp, o2):

        self.log.append({
            "step": int(step),
            "fuel": float(fuel),
            "temp": float(temp),
            "o2": float(o2),
            "setpoint": float(self.setpoint)
        })

    def save(self, path="mpc_log.json"):
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
import numpy as np
import pandas as pd
import logging

class RotaryKilnDigitalTwin:
    def __init__(self, config=None, log_level=logging.INFO, seed=None):

        if seed is not None:
            np.random.seed(seed)

        self.config = config if config else {}

        self.logger = logging.getLogger("RotaryKilnDT")
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # 🕒 TIME SCALE (NEW)
        self.step_duration_sec = self.config.get("system", {}).get("step_duration_sec", 60)

        self.reset()

    def reset(self):
        self.temp = 1450.0
        self.o2 = 3.2

        self.fuel = 18.5
        self.fan = 950.0

        self.T_env = 25.0
        self.step_count = 0
        self.data = []

        physics_cfg = self.config.get("physics", {})
        d_steps = physics_cfg.get("delayed_steps", 12)

        self.fuel_history = [self.fuel] * d_steps

        return self._get_obs()

    def _get_obs(self):
        return {
            "temp": self.temp,
            "o2": self.o2,
            "fuel": self.fuel,
            "fan": self.fan
        }

    def combustion_eff(self, o2):
        return 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.2) / 1.4) ** 2)

    def step(self, target_fuel, target_fan):

        limits = self.config.get("limits", self.config.get("plant", {}))
        f_min, f_max = limits.get("fuel_min", 12.0), limits.get("fuel_max", 28.0)
        v_min, v_max = limits.get("fan_min", 850.0), limits.get("fan_max", 1100.0)

        target_fuel = np.clip(target_fuel, f_min, f_max)
        target_fan = np.clip(target_fan, v_min, v_max)

        phys = self.config.get("physics", {})
        thermal_mass = phys.get("thermal_mass", 1050.0)
        heat_gain_f = phys.get("heat_gain_factor", 21.98)
        conv_f = phys.get("conv_factor", 0.00026)

        alpha_fan = phys.get("fan_inertia", 0.15)
        alpha_fuel = phys.get("fuel_inertia", 0.10)

        # =========================
        # FAN DYNAMICS
        # =========================
        self.fan += alpha_fan * (target_fan - self.fan)

        # =========================
        # FUEL DYNAMICS
        # =========================
        self.fuel += alpha_fuel * (target_fuel - self.fuel)

        self.fuel_history.append(self.fuel)
        delayed_fuel = self.fuel_history.pop(0)

        # =========================
        # O2 DYNAMICS
        # =========================
        fuel_effect = 1.2 * np.tanh(0.18 * (self.fuel - 16))
        fan_effect = 1.5 * np.tanh((self.fan - 1000) / 120)

        o2_target = np.clip(3.4 + fan_effect - fuel_effect, 2.0, 5.0)
        self.o2 += 0.12 * (o2_target - self.o2)

        # =========================
        # TEMPERATURE MODEL
        # =========================
        eff = self.combustion_eff(self.o2)

        heat_gain = delayed_fuel * heat_gain_f * eff
        heat_loss = (0.004 + conv_f * self.fan) * (self.temp - self.T_env)

        delta_temp = (heat_gain - heat_loss) / thermal_mass
        delta_temp = np.clip(delta_temp, -5.0, 5.0)

        self.temp += delta_temp + np.random.normal(0, 0.14)

        # =========================
        # TIME (NEW SEMANTIC LAYER)
        # =========================
        time_min = self.step_count * self.step_duration_sec / 60.0

        # =========================
        # LOG
        # =========================
        record = {
            "step": self.step_count,
            "time_min": time_min,
            "temperature": float(self.temp),
            "fuel": float(self.fuel),
            "fan": float(self.fan),
            "o2": float(self.o2),
            "efficiency": float(eff),
        }

        self.data.append(record)
        self.step_count += 1

        return record

    def run(self, steps=1440, fuel_cmd=18.5, fan_cmd=950.0):

        self.reset()

        for _ in range(steps):
            self.step(fuel_cmd, fan_cmd)

        return pd.DataFrame(self.data)
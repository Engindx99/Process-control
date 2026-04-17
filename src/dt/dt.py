import numpy as np
import pandas as pd
import pickle
import logging
import os


class RotaryKilnDigitalTwin:
    def __init__(self, seed=42, log_level=logging.INFO):

        # ---------------- LOGGING ----------------
        self.logger = logging.getLogger("RotaryKilnDT")
        self.logger.setLevel(log_level)

        if len(self.logger.handlers) == 0:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s'
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)


        # ---------------- STATE ----------------
        self.temp = 1400.0
        self.o2 = 3.0
        self.fuel = 16.0
        self.fan = 1000.0
        self.T_env = 25.0

        self.step_count = 0
        self.data = []
        self.fuel_history = [self.fuel] * 12

        # ---------------- PHYSICS ----------------
        self.thermal_mass = 100
        self.heat_gain_factor = 20.88
        self.conv_factor = 0.00022
        self.rad_factor = 5.67e-12
        self.emissivity = 0.85
        self.area_scale = 7.0

    # ---------------- O2 MODEL ----------------
    def calculate_o2_target(self, fuel, fan):
        fuel_effect = 2.2 * np.tanh(0.25 * (fuel - 16))
        fan_effect = 2.2 * np.tanh((fan - 1000) / 85)

        o2 = 3.0 + fan_effect - fuel_effect
        return np.clip(o2, 2.0, 5.0)

    # ---------------- COMBUSTION ----------------
    def combustion_eff(self, o2):
        return 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.2) / 1.4) ** 2)

    # ---------------- STEP ----------------
    def step(self, fuel, fan):

        try:
            self.fuel = np.clip(fuel, 12.0, 22.0)
            self.fan = np.clip(fan, 950.0, 1100.0)

            self.fuel_history.append(self.fuel)
            delayed_fuel = self.fuel_history.pop(0)

            # ---------------- O2 ----------------
            o2_target = self.calculate_o2_target(self.fuel, self.fan)
            self.o2 += 0.25 * (o2_target - self.o2)
            self.o2 = np.clip(self.o2, 2.0, 4.0)

            # ---------------- TEMPERATURE ----------------
            comb_eff = self.combustion_eff(self.o2)

            heat_gain = delayed_fuel * self.heat_gain_factor * comb_eff
            heat_gain = np.clip(heat_gain, 0, 2500)

            heat_loss = (0.004 + self.conv_factor * self.fan) * (self.temp - self.T_env)

            net = heat_gain - heat_loss

            # 🔥 SAFE DYNAMICS (explosion guard)
            delta_temp = self.thermal_mass * (net / 1000.0)
            delta_temp = np.clip(delta_temp, -50, 50)

            self.temp += delta_temp

            # ---------------- EFFICIENCY ----------------
            temp_eff = np.exp(-0.5 * ((self.temp - 1450) / 80) ** 2)
            o2_eff = np.exp(-0.5 * ((self.o2 - 3.0) / 0.5) ** 2)

            eff = np.clip(0.6 * temp_eff + 0.4 * o2_eff, 0.0, 1.0)

            # ---------------- LOG ----------------
            record = {
                "Step": self.step_count,
                "Fuel": float(self.fuel),
                "Fan": float(self.fan),
                "Temperature": float(self.temp),
                "O2": float(self.o2),
                "Efficiency": float(eff)
            }

            self.data.append(record)
            self.step_count += 1

            return record

        except Exception as e:
            self.logger.error(f"Step error: {e}")
            raise

    # ---------------- SIMULATION ----------------
    def run_full_simulation(self, steps=3000, save_csv=True, csv_path="kiln_dataset.csv"):

        self.logger.info("Simulation started")

        f_val, v_val = 16.0, 1000.0

        for _ in range(steps):
            f_val = np.clip(f_val + np.random.normal(0, 0.1), 13, 21)
            v_val = np.clip(v_val + np.random.normal(0, 2.0), 960, 1080)
            self.step(f_val, v_val)

        df = pd.DataFrame(self.data)

        if save_csv:
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Dataset saved -> {csv_path}")

        self.logger.info(
            f"T range: {df['Temperature'].min():.0f} - {df['Temperature'].max():.0f}"
        )
        self.logger.info(
            f"O2 range: {df['O2'].min():.2f} - {df['O2'].max():.2f}"
        )

        return df

    # ---------------- SAFE CHECKPOINT ----------------
    def save_checkpoint(self, path="dt_checkpoint.pkl"):

        state = {
            "temp": self.temp,
            "o2": self.o2,
            "fuel": self.fuel,
            "fan": self.fan,
            "T_env": self.T_env,
            "step_count": self.step_count,
            "data": self.data,
            "fuel_history": self.fuel_history
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        self.logger.info(f"Checkpoint saved -> {path}")

    def load_checkpoint(self, path="dt_checkpoint.pkl"):

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        for k, v in state.items():
            setattr(self, k, v)

        self.logger.info(f"Checkpoint loaded -> {path}")
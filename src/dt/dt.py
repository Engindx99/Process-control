import numpy as np
import pandas as pd
import logging

class RotaryKilnDigitalTwin:
    def __init__(self, config=None, log_level=logging.INFO, seed=None):
        # ---------------- SEEDING (MPC ve Eğitim Tekrarlanabilirliği İçin) ----------------
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

        # ---------------- CONFIG MANAGEMENT ----------------
        self.config = config if config else {
            "physics": {
                "thermal_mass": 3000,        
                "heat_gain_factor": 20.0, 
                "conv_factor": 0.00020,      # Stabil soğuma katsayısı
                "delayed_steps": 12,
                "fan_inertia": 0.15 
            },
            "noise": {
                "fuel_std": 0.05,
                "fan_std": 0.3,
                "fan_drift_scale": 5
            },
            "limits": {
                "temp_delta_max": 5.0,
                "setpoint": 1450.0
            }
        }
        
        # ---------------- LOGGING ----------------
        self.logger = logging.getLogger("RotaryKilnDT")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        self.reset()

    def reset(self):
        self.temp = 1400.0
        self.o2 = 3.2 
        self.fuel = 16.0
        self.fan = 1000.0
        self.T_env = 25.0
        self.step_count = 0
        self.data = []
        self.fuel_history = [self.fuel] * self.config["physics"]["delayed_steps"]
        return self._get_obs()

    def _get_obs(self):
        return {"temp": self.temp, "o2": self.o2, "fuel": self.fuel, "fan": self.fan}

    def combustion_eff(self, o2):
        return 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.2) / 1.4) ** 2)

    def step(self, target_fuel, target_fan):
        try:
            target_fuel = np.clip(target_fuel, 12.0, 22.0)
            target_fan = np.clip(target_fan, 850.0, 1100.0)

            # 1. FAN ATALETİ
            alpha = self.config["physics"].get("fan_inertia", 0.15)
            self.fan += alpha * (target_fan - self.fan)

            # 2. YAKIT GECİKMESİ
            self.fuel = target_fuel
            self.fuel_history.append(self.fuel)
            delayed_fuel = self.fuel_history.pop(0)

            # 3. O2 DİNAMİĞİ
            fuel_effect = 1.2 * np.tanh(0.18 * (self.fuel - 16))
            fan_effect = 1.5 * np.tanh((self.fan - 1000) / 120)
            
            o2_target = np.clip(3.4 + fan_effect - fuel_effect, 2.0, 5.0)
            self.o2 += 0.12 * (o2_target - self.o2)

            # 4. ISI DENGESİ
            eff = self.combustion_eff(self.o2)
            heat_gain = delayed_fuel * self.config["physics"]["heat_gain_factor"] * eff
            heat_loss = (0.004 + self.config["physics"]["conv_factor"] * self.fan) * (self.temp - self.T_env)
            
            delta_temp = (heat_gain - heat_loss) / self.config["physics"]["thermal_mass"]
            delta_temp = np.clip(delta_temp, -5.0, 5.0)
            
            self.temp += delta_temp + np.random.normal(0, 0.4)

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
            self.logger.error(f"Step Error: {e}")
            raise

    def run_full_simulation(self, steps=3000):
        self.logger.info(f"Simulating {steps} steps with balanced Fuel-O2.")
        f_val = 16.0
        
        for i in range(steps):
            f_val = np.clip(f_val + np.random.normal(0, self.config["noise"]["fuel_std"]), 13, 21)
            
            fuel_demand = (f_val - 16) * 6
            temp_demand = (self.temp - 1420) * 1.0
            
            v_target = 1000 + fuel_demand + temp_demand
            v_val = np.clip(v_target + np.random.normal(0, 1.0), 900, 1100)
            
            self.step(f_val, v_val)

        return pd.DataFrame(self.data)
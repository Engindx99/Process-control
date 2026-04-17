import numpy as np
import pandas as pd
import pickle
import logging
import os

class RotaryKilnDigitalTwin:
    def __init__(self, config=None, log_level=logging.INFO):
        # ---------------- CONFIG MANAGEMENT (MLOps Pillar) ----------------
        self.config = config if config else {
            "physics": {
                "thermal_mass": 2600,
                "heat_gain_factor": 21.7,
                "conv_factor": 0.00022,
                "delayed_steps": 12,
                "fan_inertia": 0.15  # Atalet katsayısı (0.1: Çok ağır, 1.0: Anlık tepki)
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

        # ---------------- STATE ----------------
        self.reset()

    def reset(self):
        """Simülasyonu başlangıç durumuna döndürür."""
        self.temp = 1400.0
        self.o2 = 3.0
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
        """
        target_fuel: Kontrolcüden gelen hedef yakıt miktarı
        target_fan: Kontrolcüden gelen hedef fan hızı
        """
        try:
            # 1. Giriş Sınırlandırma (Actuator Limits)
            target_fuel = np.clip(target_fuel, 12.0, 22.0)
            target_fan = np.clip(target_fan, 950.0, 1100.0)

            # 2. FAN ATALETİ (First Order Lag)
            # Mevcut fan hızı, hedef hıza atalet katsayısı kadar yaklaşır.
            # Bu, fanın aniden hızlanmasını engeller.
            alpha = self.config["physics"].get("fan_inertia", 0.15)
            self.fan += alpha * (target_fan - self.fan)

            # 3. YAKIT GECİKMESİ
            self.fuel = target_fuel
            self.fuel_history.append(self.fuel)
            delayed_fuel = self.fuel_history.pop(0)

            # 4. O2 DİNAMİĞİ (O2 tepkisi fanın o anki hızına bağlıdır)
            fuel_effect = 2.2 * np.tanh(0.25 * (self.fuel - 16))
            fan_effect = 2.2 * np.tanh((self.fan - 1000) / 85)
            o2_target = np.clip(3.0 + fan_effect - fuel_effect, 2.0, 5.0)
            self.o2 += 0.25 * (o2_target - self.o2)

            # 5. ISI DENGESİ
            eff = self.combustion_eff(self.o2)
            heat_gain = delayed_fuel * self.config["physics"]["heat_gain_factor"] * eff
            heat_loss = (0.004 + self.config["physics"]["conv_factor"] * self.fan) * (self.temp - self.T_env)
            
            delta_temp = (heat_gain - heat_loss) / self.config["physics"]["thermal_mass"]
            delta_temp = np.clip(delta_temp, -self.config["limits"]["temp_delta_max"], 
                                          self.config["limits"]["temp_delta_max"])
            
            self.temp += delta_temp + np.random.normal(0, 0.08)

            record = {
                "Step": self.step_count,
                "Fuel": float(self.fuel),
                "Fan": float(self.fan),
                "Temperature": float(self.temp),
                "O2": float(self.o2)
            }
            self.data.append(record)
            self.step_count += 1
            return record

        except Exception as e:
            self.logger.error(f"Step Error: {e}")
            raise

    def run_full_simulation(self, steps=3000):
        """Atalet ve gürültü içeren gerçekçi veri seti üretimi."""
        self.logger.info(f"Simulating {steps} steps.")
        f_val, v_val = 16.0, 1000.0
        
        for i in range(steps):
            # Yakıt ve Fan gürültüsü
            f_val = np.clip(f_val + np.random.normal(0, self.config["noise"]["fuel_std"]), 13, 21)
            
            drift = self.config["noise"]["fan_drift_scale"] * np.sin(i / 150)
            v_val = np.clip(1000 + drift + np.random.normal(0, self.config["noise"]["fan_std"]), 960, 1080)
            
            self.step(f_val, v_val)

        return pd.DataFrame(self.data)
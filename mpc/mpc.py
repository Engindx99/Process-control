"""
MPC Controller Sınıfı - Sabit Setpoint 1450°C
"""

import numpy as np
import json
from digital_twin.dt import RotaryKilnDigitalTwin


class MPC:
    """Model Predictive Control - Sabit Setpoint: 1450°C"""
    
    def __init__(self, horizon=10, n_candidates=12):
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.lambda_u = 0.6  # Yakıt değişim cezası
        self.setpoint = 1450  # 📌 SABİT SETPOINT
        self.log = []

    def cost(self, temps, fuels):
        """Maliyet fonksiyonu - sabit setpoint kullanır"""
        error = np.array(temps) - self.setpoint
        du = np.diff(fuels, prepend=fuels[0])
        return np.sum(error**2) + self.lambda_u * np.sum(du**2)

    def optimize(self, plant):
        """Optimum yakıt debisini bul - setpoint parametresi gerekmez"""
        best_fuel = plant.fuel
        best_cost = float("inf")
        
        fuel_grid = np.linspace(12.0, 22.0, self.n_candidates)

        for f in fuel_grid:
            sim = self._clone(plant)
            temps = []
            fuels = []

            for _ in range(self.horizon):
                # Model belirsizliği
                sim.temp += np.random.normal(0, 0.3)
                temp, o2, _ = sim.step(f, 1000.0)
                temps.append(temp)
                fuels.append(f)

            c = self.cost(temps, fuels)

            if c < best_cost:
                best_cost = c
                best_fuel = f

        return best_fuel

    def _clone(self, plant):
        """Plant kopyala"""
        sim = RotaryKilnDigitalTwin()
        sim.temp = plant.temp
        sim.o2 = plant.o2
        sim.fuel = plant.fuel
        sim.fan = plant.fan
        sim.fuel_history = plant.fuel_history.copy()
        return sim

    def log_step(self, step, fuel, temp, o2):
        """Adımı logla - setpoint otomatik eklenir"""
        self.log.append({
            "step": step,
            "fuel": float(fuel),
            "temp": float(temp),
            "o2": float(o2),
            "setpoint": float(self.setpoint)  # Sabit setpoint
        })

    def save(self, path="mpc_log.json"):
        """Log'u kaydet"""
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
    
    def load(self, path="mpc_log.json"):
        """Log'u yükle"""
        with open(path, "r") as f:
            self.log = json.load(f)
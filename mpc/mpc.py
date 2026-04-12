import numpy as np
import json
from scipy.optimize import minimize_scalar

class MPC:
    def __init__(self, horizon=20):
        self.horizon = horizon
        self.lambda_u = 15.0  # Hareket maliyeti
        self.setpoint = 1450
        # Yakıtın sadece 0.02'den büyük değişimlerini "gerçek değişim" say
        self.deadband = 0.02 
        self.log = []

    def _objective(self, f, plant):
        """Maliyet fonksiyonu: Simülasyon sonundaki hata karesi + değişim cezası"""
        sim = self._clone(plant)
        temps = []
        for _ in range(self.horizon):
            temp, _, _ = sim.step(f, 1000.0)
            temps.append(temp)
        
        mse = np.mean((np.array(temps) - self.setpoint)**2)
        # Mevcut yakıttan uzaklaşma cezası (stabilite sağlar)
        smoothness = self.lambda_u * (f - plant.fuel)**2
        return mse + smoothness

    def optimize(self, plant):
        # Aday listesi kullanmak yerine matematiksel minimumu buluyoruz
        # Bu sayede 17.39 veya 17.56 yerine 17.4234 gibi tam gereken değeri bulur.
        res = minimize_scalar(
            self._objective, 
            args=(plant,), 
            bounds=(12.0, 22.0), 
            method='bounded'
        )
        
        new_fuel = res.x
        
        # --- Ölü Bant (Deadband) Mantığı ---
        # Eğer yeni bulunan değer, eskisine çok yakınsa boşuna vanayı oynatma.
        if abs(new_fuel - plant.fuel) < self.deadband:
            return plant.fuel
            
        return new_fuel

    def _clone(self, plant):
        from digital_twin.dt import RotaryKilnDigitalTwin
        sim = RotaryKilnDigitalTwin()
        sim.temp = plant.temp
        sim.o2 = plant.o2
        sim.fuel = plant.fuel
        sim.fan = plant.fan
        sim.fuel_history = plant.fuel_history.copy()
        return sim

    def log_step(self, step, fuel, temp, o2):
        # Sadece sayısal değişim varsa logla diyebilirsin ama 
        # grafik sürekliliği için her adımı yazmak daha iyidir.
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
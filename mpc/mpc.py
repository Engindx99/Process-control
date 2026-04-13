import numpy as np
import json
from scipy.optimize import minimize
from digital_twin.dt import RotaryKilnDigitalTwin

class MPC:
    def __init__(self, prediction_horizon=30, control_horizon=3):
        """
        Gelişmiş Vektörel MPC Kontrolcüsü
        :param prediction_horizon (P): Sistemin gelecekteki tepkisini izleme süresi
        :param control_horizon (M): Kaç adım sonrasına kadar karar alınacağı
        """
        self.P = prediction_horizon
        self.M = control_horizon
        self.setpoint = 1450
        
        # PARAMETRE AYARLARI
        self.lambda_u = 120.0   # Yakıt değişim cezası (Vektörel yapıda biraz daha yüksek tutulabilir)
        self.max_step_change = 0.05 # Slew rate limiti
        
        self.log = []

    def _objective(self, fuel_sequence, plant):
        """Maliyet fonksiyonu: Sıcaklık hatası + Kontrol çabası"""
        sim = self._clone(plant)
        temps = []
        
        # Kontrol ufku dışındaki adımlar için son kontrol değerini kullan (Zero-order hold)
        full_sequence = np.zeros(self.P)
        full_sequence[:self.M] = fuel_sequence
        full_sequence[self.M:] = fuel_sequence[-1]
        
        # Simülasyonu P adımı boyunca koştur
        for i in range(self.P):
            temp, _, _ = sim.step(full_sequence[i], 1000.0)
            temps.append(temp)
        
        # 1. Hataların karesi (Setpoint takibi)
        mse = np.mean((np.array(temps) - self.setpoint)**2)
        
        # 2. Kontrol Çabası (Süreklilik/Düzgünleştirme)
        # Mevcut yakıtla ilk önerilen yakıt arasındaki ve önerilenlerin kendi arasındaki farkı cezalandır
        diffs = np.diff(np.insert(fuel_sequence, 0, plant.fuel))
        smoothness = self.lambda_u * np.sum(diffs**2)
        
        return mse + smoothness

    def optimize(self, plant):
        # Başlangıç tahmini: Mevcut yakıt değerini M adım boyunca koru
        initial_guess = np.full(self.M, plant.fuel)
        
        # Yakıt sınırları (Constraint)
        bounds = [(12.0, 22.0) for _ in range(self.M)]
        
        # Optimizasyon (SLSQP kısıtlı optimizasyon için uygundur)
        res = minimize(
            self._objective, 
            initial_guess, 
            args=(plant,), 
            bounds=bounds, 
            method='SLSQP'
        )
        
        # Sadece ilk adımı seç (Receding Horizon prensibi)
        target_fuel = res.x[0]
        
        # --- FİZİKSEL KISIT: Slew Rate (Hız Limiti) ---
        current_fuel = plant.fuel
        diff = target_fuel - current_fuel
        
        if abs(diff) > self.max_step_change:
            actual_fuel = current_fuel + np.sign(diff) * self.max_step_change
        else:
            actual_fuel = target_fuel
            
        # Histerezis / Deadband (0.01 altındaki oynamalar vana ömrü için engellenir)
        if abs(actual_fuel - current_fuel) < 0.01:
            return current_fuel
            
        return actual_fuel

    def _clone(self, plant):
        """Digital Twin kopyası oluşturur"""
        
        sim = RotaryKilnDigitalTwin()
        # Durum transferi
        sim.temp = plant.temp
        sim.o2 = plant.o2
        sim.fuel = plant.fuel
        sim.fan = plant.fan
        if hasattr(plant, 'fuel_history'):
            sim.fuel_history = plant.fuel_history.copy()
        return sim

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
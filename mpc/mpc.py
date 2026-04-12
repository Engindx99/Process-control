import numpy as np
import json
from scipy.optimize import minimize_scalar

class MPC:
    def __init__(self, horizon=20):
        """
        Gelişmiş MPC Kontrolcüsü
        :param horizon: Geleceğe yönelik tahmin adımı (Gecikme toleransı için 20 idealdir)
        """
        self.horizon = horizon
        self.setpoint = 1450
        
        # PARAMETRE AYARLARI (Sakinleştirilmiş Kontrol)
        self.lambda_u = 80.0    # Yakıt değişim cezası (Yüksek değer = Daha düz mavi hat)
        self.deadband = 0.08    # Ölü bant (Bu değerden küçük değişimler yok sayılır)
        
        self.log = []

    def _objective(self, f, plant):
        """Maliyet fonksiyonu: Sıcaklık hatası + Yakıt değişim maliyeti"""
        sim = self._clone(plant)
        temps = []
        
        # Belirlenen ufuk (horizon) boyunca simülasyon yap
        for _ in range(self.horizon):
            temp, _, _ = sim.step(f, 1000.0)
            temps.append(temp)
        
        # 1. Sıcaklık hatasının karesi (MSE)
        mse = np.mean((np.array(temps) - self.setpoint)**2)
        
        # 2. Yakıt değişiminin maliyeti (Düzgünleştirme etkisi)
        smoothness = self.lambda_u * (f - plant.fuel)**2
        
        return mse + smoothness

    def optimize(self, plant):
        res = minimize_scalar(
            self._objective, 
            args=(plant,), 
            bounds=(12.0, 22.0), 
            method='bounded'
        )
        
        target_fuel = res.x
        
        # --- FİZİKSEL KISIT: Slew Rate (Hız Limiti) ---
        # Yakıtın bir adımda en fazla 0.05 birim değişebileceğini varsayalım
        max_step_change = 0.05 
        current_fuel = plant.fuel
        
        diff = target_fuel - current_fuel
        
        if abs(diff) > max_step_change:
            # Eğer değişim çok büyükse, vana sadece kapasitesi kadar döner
            actual_fuel = current_fuel + np.sign(diff) * max_step_change
        else:
            actual_fuel = target_fuel
            
        # --- Histerezis (Yine de Deadband'i koruyalım) ---
        if abs(actual_fuel - current_fuel) < 0.01:
            return current_fuel
            
        return actual_fuel

    def _clone(self, plant):
        """Digital Twin'in o anki durumunu (ve geçmişini) kopyalar"""
        from digital_twin.dt import RotaryKilnDigitalTwin
        sim = RotaryKilnDigitalTwin()
        sim.temp = plant.temp
        sim.o2 = plant.o2
        sim.fuel = plant.fuel
        sim.fan = plant.fan
        sim.fuel_history = plant.fuel_history.copy()
        return sim

    def log_step(self, step, fuel, temp, o2):
        """Her adımın sonucunu hafızaya kaydeder"""
        self.log.append({
            "step": int(step),
            "fuel": float(fuel),
            "temp": float(temp),
            "o2": float(o2),
            "setpoint": float(self.setpoint)
        })

    def save(self, path="mpc_log.json"):
        """Log verilerini JSON dosyasına yazar"""
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)

    def load(self, path="mpc_log.json"):
        """Eski log verilerini yükler"""
        with open(path, "r") as f:
            self.log = json.load(f)
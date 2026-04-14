import numpy as np
import json
from scipy.optimize import minimize
from src.digital_twin.dt import RotaryKilnDigitalTwin

class MPC:
    def __init__(self, prediction_horizon=30, control_horizon=3):
        self.P = prediction_horizon  # Geleceği görme ufku
        self.M = control_horizon     # Kontrol hamlesi ufku
        self.setpoint = 1450.0

        # =============================================================
        # 1. COST WEIGHTS (AGRESİFLİK AYARI)
        # =============================================================
        # Yakıt değişim cezası (Smoothness). 120'den 800'e çıkardık.
        # Bu değer ne kadar yüksekse, MPC o kadar "sakin" davranır.
        self.lambda_u = 800.0 
        
        # O2 dengesi cezası
        self.lambda_o2 = 15.0 

        # logging
        self.log = []

    def _objective(self, fuel_sequence, plant):
        sim = self._clone(plant)
        
        temps = []
        o2_values = []

        # =============================================================
        # 2. RL BIAS (HIBRIT ZEKA ENTEGRASYONU)
        # =============================================================
        # MPC simülasyon yaparken, RL'in son yaptığı müdahaleyi hesaba katar.
        # Bu, MPC'nin RL'i bir "bozucu etki" sanıp panik yapmasını engeller.
        rl_bias = getattr(plant, 'last_rl_residual', 0.0)

        # Kontrol ufku dışındaki adımlar için son yakıt değerini koru
        full_sequence = np.zeros(self.P)
        full_sequence[:self.M] = fuel_sequence
        full_sequence[self.M:] = fuel_sequence[-1]

        for i in range(self.P):
            # Simülasyonda RL etkisini de ekliyoruz
            temp, o2, _ = sim.step(full_sequence[i] + rl_bias, 1000.0)
            temps.append(temp)
            o2_values.append(o2)

        temps = np.array(temps)
        o2_values = np.array(o2_values)

        # --- HATA HESABI (MSE) ---
        # Sıcaklık farklarını 10'a bölerek normalize ediyoruz (Sayısall kararlılık)
        mse = np.mean(((temps - self.setpoint) / 10.0) ** 2)

        # --- O2 CEZASI ---
        o2_penalty = self.lambda_o2 * np.mean((o2_values - 3.0) ** 2)

        # --- SMOOTHNESS (DEĞİŞİM CEZASI) ---
        # Mevcut yakıt ile planlanan yakıt arasındaki farkların karesi
        diffs = np.diff(np.concatenate([[plant.fuel], fuel_sequence]))
        smoothness = self.lambda_u * np.sum(diffs ** 2)

        # --- ENERJİ VERİMLİLİĞİ ---
        energy_penalty = 0.05 * np.mean(fuel_sequence ** 2)

        return mse + smoothness + o2_penalty + energy_penalty

    def optimize(self, plant):
        # Mevcut yakıt miktarından başla
        initial_guess = np.full(self.M, plant.fuel)

        # Fiziksel yakıt sınırları (m3/h)
        bounds = [(12.0, 22.0) for _ in range(self.M)]

        # Optimizasyon (SLSQP algoritması)
        res = minimize(
            self._objective,
            initial_guess,
            args=(plant,),
            bounds=bounds,
            method='SLSQP',
            options={'ftol': 1e-3} # Çok küçük değişimleri görmezden gel (hız için)
        )

        target_fuel = res.x[0]
        current_fuel = plant.fuel

        # =============================================================
        # 3. SLEW RATE LIMITER (ZIKZAK ENGELLEYİCİ)
        # =============================================================
        # MPC'nin bir adımda yakıtı en fazla ne kadar değiştirebileceğini sabitliyoruz.
        # Bu, sistemin mekanik ataletine uyum sağlar ve salınımı (oscillation) keser.
        max_allowed_change = 0.05 
        
        diff = target_fuel - current_fuel

        if abs(diff) > max_allowed_change:
            actual_fuel = current_fuel + np.sign(diff) * max_allowed_change
        else:
            actual_fuel = target_fuel

        # Ölü Bölge (Deadzone): Çok küçük değişimler için aktüatörü yorma
        if abs(actual_fuel - current_fuel) < 0.005:
            return current_fuel

        return actual_fuel

    def _clone(self, plant):
        """Fiziksel modelin anlık kopyasını oluşturur (Simülasyon için)"""
        sim = RotaryKilnDigitalTwin()
        sim.temp = plant.temp
        sim.o2 = plant.o2
        sim.fuel = plant.fuel
        sim.fan = plant.fan
        return sim

    def log_step(self, step, fuel, temp, o2):
        self.log.append({
            "step": int(step),
            "fuel": float(fuel),
            "temp": float(temp),
            "o2": float(o2),
            "setpoint": float(self.setpoint)
        })

    def save(self, path="data/mpc_log.json"):
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
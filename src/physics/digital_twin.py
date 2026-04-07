# digital_twin.py
import numpy as np

class AdvancedRotaryKiln:
    def __init__(self):
        # --- Sabitler ---
        self.mass = 5000.0
        self.cp = 1.1
        self.k_conv = 0.04
        self.k_rad = 1e-10   # radyasyon (T^4 etkisi)
        self.ambient_temp = 25.0
        self.max_temp = 1550
        self.min_temp = 800

    def combustion_efficiency(self, oxygen_level, fan_speed):
        """O2 ve fan etkisine göre yanma verimi (0.6 - 1.0 arası)"""
        o2_eff = np.exp(-0.1 * (oxygen_level - 4)**2)
        # Fan verimi (tanh ile yumuşatılmış)
        fan_eff = np.tanh(fan_speed / 50)
        return 0.6 + 0.4 * o2_eff * fan_eff

    def get_dynamics(self, temp, oxygen, fuel, feed, fan):
        """
        MPC ve RL için türev (değişim) hesaplar.
        Not: CasADi sembolik değişkenleriyle uyumlu olması için 
        numpy fonksiyonlarını (np.exp, np.tanh) kullanırken dikkat edilmelidir.
        """
        # 1. Yanma verimi
        eta = self.combustion_efficiency(oxygen, fan)

        # 2. Isı dengesi bileşenleri
        q_in = fuel * 30000 * eta
        q_loss = self.k_conv * (temp - self.ambient_temp) + self.k_rad * (temp**4 - self.ambient_temp**4)
        q_material = feed * self.cp * (temp - 100)
        q_fan_cooling = 0.02 * fan * (temp - self.ambient_temp)

        # 3. Diferansiyeller
        dT_dt = (q_in - q_loss - q_material - q_fan_cooling) / (self.mass * self.cp)
        dO2_dt = 0.01 * (fan - 50) - 0.005 * fuel

        return dT_dt, dO2_dt
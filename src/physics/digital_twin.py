import numpy as np

class AdvancedRotaryKiln:
    def __init__(self):
        # --- Sabitler ---
        self.mass = 5000.0
        self.cp = 1.1

        # Heat loss coefficients
        self.k_conv = 0.04
        self.k_rad = 1e-10   # radyasyon (T^4 etkisi)

        self.ambient_temp = 25.0

        # --- State ---
        self.current_temp = 1200.0
        self.oxygen_level = 3.0

        # --- Limits ---
        self.max_temp = 1500
        self.min_temp = 800

    # -----------------------------
    # 🔥 Yanma verimi modeli
    # -----------------------------
    def combustion_efficiency(self, oxygen_level, fan_speed):
        """
        O2 ve fan etkisine göre yanma verimi (0-1 arası)
        """
        # O2 optimum ~3-5%
        o2_eff = np.exp(-0.1 * (oxygen_level - 4)**2)

        # Fan çok düşükse yanma kötü, çok yüksekse cooling
        fan_eff = np.tanh(fan_speed / 50)

        return 0.6 + 0.4 * o2_eff * fan_eff

    # -----------------------------
    # 🌡️ Dinamik model
    # -----------------------------
    def step(self, fuel_rate, material_feed, fan_speed):
        # --- 1. Yanma verimi ---
        eta = self.combustion_efficiency(self.oxygen_level, fan_speed)

        # --- 2. Isı girişi ---
        q_in = fuel_rate * 30000 * eta

        # --- 3. Isı kaybı (nonlinear) ---
        T = self.current_temp
        Ta = self.ambient_temp

        q_loss = self.k_conv * (T - Ta) + self.k_rad * (T**4 - Ta**4)

        # --- 4. Hammadde etkisi ---
        feed_temp = 100
        q_material = material_feed * self.cp * (T - feed_temp)

        # --- 5. Fan ile ekstra soğutma ---
        q_fan_cooling = 0.02 * fan_speed * (T - Ta)

        # --- 6. Diferansiyel ---
        dT = (q_in - q_loss - q_material - q_fan_cooling) / (self.mass * self.cp)

        # --- 7. Zaman entegrasyonu ---
        dt = 1.0
        self.current_temp += dT * dt

        # --- 8. Gürültü ---
        noise = np.random.normal(0, 0.3)
        self.current_temp += noise

        # --- 9. Fiziksel sınırlar ---
        self.current_temp = np.clip(self.current_temp, self.min_temp, self.max_temp)

        # --- 10. O2 dinamiği (basit model) ---
        self.oxygen_level += 0.01 * (fan_speed - 50) - 0.005 * fuel_rate
        self.oxygen_level = np.clip(self.oxygen_level, 1, 10)

        return self.current_temp, self.oxygen_level

    def reset(self):
        self.current_temp = 1200.0
        self.oxygen_level = 3.0
        return self.current_temp, self.oxygen_level
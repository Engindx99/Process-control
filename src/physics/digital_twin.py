import numpy as np

class IndustrialRotaryKiln:
    def __init__(self, dt=1.0):
        # --- Zaman ---
        self.dt = dt

        # --- Fiziksel Sabitler (İyileştirildi) ---
        self.mass = 5000.0
        self.cp = 1.1
        
        # Isı kaybı katsayıları artırıldı (Fırının soğuyabilmesi için)
        self.k_conv = 0.08  # 0.04 -> 0.08
        self.k_rad = 5e-10  # 1e-10 -> 5e-10
        self.ambient_temp = 25.0

        # --- Limitler ---
        self.max_temp = 1600 # Tavana çarpmayı önlemek için biraz yükseltildi
        self.min_temp = 700

        # --- Actuator state ---
        self.fuel_actual = 0.0
        self.fan_actual = 50.0

        # --- Delay buffer ---
        self.delay_steps = 5
        self.fuel_history = [0.0] * self.delay_steps
        self.fan_history = [50.0] * self.delay_steps

        # --- State ---
        self.temp = 900.0
        self.oxygen = 3.0

    # -----------------------------
    # 🔥 Yanma verimi (Geliştirildi)
    # -----------------------------
    def combustion_efficiency(self, oxygen_level, fan_speed):
        # O2 seviyesi %4'ten uzaklaştıkça verim düşer
        o2_eff = np.exp(-0.1 * (oxygen_level - 4)**2)
        fan_eff = np.tanh(fan_speed / 40) # Düşük fan hızlarında bile daha stabil verim
        return 0.55 + 0.45 * o2_eff * fan_eff

    # -----------------------------
    # ⚙️ Actuator dynamics (Inertia)
    # -----------------------------
    def apply_actuator(self, fuel_cmd, fan_cmd):
        # Yakıt ve fan değişim hızları
        fuel_rate = 0.8 # Tepki hızı biraz artırıldı
        fan_rate = 3.0

        self.fuel_actual += np.clip(fuel_cmd - self.fuel_actual, -fuel_rate, fuel_rate)
        self.fan_actual += np.clip(fan_cmd - self.fan_actual, -fan_rate, fan_rate)

        return self.fuel_actual, self.fan_actual

    # -----------------------------
    # 🌪️ Disturbance
    # -----------------------------
    def apply_disturbance(self, feed):
        # Beslemedeki %10'luk dalgalanma
        feed_variation = feed * np.random.uniform(0.9, 1.1)
        return feed_variation

    # -----------------------------
    # 🔁 Delay sistemi
    # -----------------------------
    def apply_delay(self, fuel, fan):
        self.fuel_history.append(fuel)
        self.fan_history.append(fan)

        fuel_delayed = self.fuel_history.pop(0)
        fan_delayed = self.fan_history.pop(0)

        return fuel_delayed, fan_delayed

    # -----------------------------
    # 🔬 Sistem dinamiği (Termal Denge)
    # -----------------------------
    def dynamics(self, temp, oxygen, fuel, feed, fan):
        eta = self.combustion_efficiency(oxygen, fan)

        # Enerji Girişi (Yanma)
        q_in = fuel * 32000 * eta # Isıl değer hafif artırıldı
        
        # Isı Kayıpları (Konveksiyon + Radyasyon)
        q_loss = self.k_conv * (temp - self.ambient_temp) \
               + self.k_rad * (temp**4 - self.ambient_temp**4)

        # Malzeme Isınma Yükü
        q_material = feed * self.cp * (temp - 100)
        
        # 🔥 Fan Soğutma Etkisi (Güçlendirildi!)
        # Fan artık fırını çok daha etkili soğutuyor
        q_fan_cooling = 0.08 * fan * (temp - self.ambient_temp)

        # Diferansiyel Denklemler
        dT_dt = (q_in - q_loss - q_material - q_fan_cooling) / (self.mass * self.cp)
        
        # Oksijen Dengesi: Fan havayı (O2) artırır, yakıt tüketir
        dO2_dt = 0.010 * fan - 0.12 * fuel

        return dT_dt, dO2_dt

    # -----------------------------
    # 🎲 Sensor modeli (Noise)
    # -----------------------------
    def measure(self):
        temp_meas = self.temp + np.random.normal(0, 1.5)
        o2_meas = self.oxygen + np.random.normal(0, 0.05)
        return temp_meas, o2_meas

    # -----------------------------
    # 🚀 STEP
    # -----------------------------
    def step(self, fuel_cmd, fan_cmd, feed):
        # 1. Aktüatör dinamiği
        fuel, fan = self.apply_actuator(fuel_cmd, fan_cmd)

        # 2. Taşıma gecikmesi (Delay)
        fuel_d, fan_d = self.apply_delay(fuel, fan)

        # 3. Besleme dalgalanması
        feed_real = self.apply_disturbance(feed)

        # 4. Dinamik hesaplama
        dT_dt, dO2_dt = self.dynamics(
            self.temp,
            self.oxygen,
            fuel_d,
            feed_real,
            fan_d
        )

        # 5. Durum güncelleme (Euler integration)
        self.temp += self.dt * dT_dt
        self.oxygen += self.dt * dO2_dt

        # 6. Güvenli limitlere çekme
        self.temp = np.clip(self.temp, self.min_temp, self.max_temp)
        self.oxygen = np.clip(self.oxygen, 0, 10)

        # 7. Gürültülü ölçüm
        temp_meas, o2_meas = self.measure()

        return {
            "temp": temp_meas,
            "oxygen": o2_meas,
            "true_temp": self.temp,
            "true_oxygen": self.oxygen,
            "fuel": fuel,
            "fan": fan
        }
import numpy as np

# =========================================================
# 1. YARDIMCI ARAÇLAR (Gürültü ve Akışkanlaştırma)
# =========================================================
class NoiseModel:
    def __init__(self):
        self.drift_raw = {k: 0.0 for k in ["T1","T2","T3","T4"]}
        self.drift_smooth = {k: 0.0 for k in ["T1","T2","T3","T4"]}
        self.state = {k: 0.0 for k in ["T1","T2","T3","T4"]}

    def get_smooth_drift(self, key, intensity=0.01):
        """Eğrilerdeki keskin dönüşleri engelleyen filtre"""
        self.drift_raw[key] = (self.drift_raw[key] * 0.999) + np.random.normal(0, intensity)
        self.drift_smooth[key] = (self.drift_smooth[key] * 0.997) + (self.drift_raw[key] * 0.003)
        return self.drift_smooth[key]

    def process_noise(self, key, sigma):
        """Sensördeki hafif titreşim/çıtırtı efekti"""
        eps = np.random.normal(0, sigma)
        self.state[key] = 0.95 * self.state[key] + eps
        return self.state[key]

# =========================================================
# 2. FİZİKSEL BİLEŞENLER (Energy Nodes & Actuators)
# =========================================================
class KilnZone:
    def __init__(self, T0, inertia=1.0):
        self.T = T0
        self.inertia = inertia 

    def update(self, Q_in, T_up, loss):
        # 🔥 İletim ve Işınım hesapları
        conduction = 0.65 * (T_up - self.T)
        rad = 3e-11 * ((self.T + 273)**4 - 298**4)

        # Enerji Dengesi + Güçlendirilmiş Atalet
        dT = (Q_in + conduction - loss * (self.T - 25) - rad) * (1.2e-5 * self.inertia)
        dT += 0.001 * np.sin(self.T / 30) # Küçük kararsızlık

        self.T += dT
        return self.T

class ActuatorModel:
    def __init__(self):
        self.fuel, self.fan = 18.5, 950.0
        self.fuel_buf = [18.5]*80 # Valf gecikmesi
        self.fan_buf = [950.0]*40  # Fan ataleti

    def step(self, f_cmd, n_cmd):
        self.fuel_buf.append(np.clip(f_cmd, 0, 40))
        self.fan_buf.append(np.clip(n_cmd, 500, 1200))
        f, n = self.fuel_buf.pop(0), self.fan_buf.pop(0)
        self.fuel += 0.005 * (f - self.fuel)
        self.fan  += 0.01 * (n - self.fan)
        return self.fuel, self.fan

# =========================================================
# 3. ANA SİSTEM (Rotary Kiln Plant)
# =========================================================
class RotaryKilnPlant:
    def __init__(self):
        self.act = ActuatorModel()
        self.noise = NoiseModel()

        # Atalet Tanımları (Inertia düştükçe kütle ağırlaşır)
        self.z1 = KilnZone(850.0,  inertia=0.4)   
        self.z2 = KilnZone(1100.0, inertia=0.2)   
        self.z3 = KilnZone(1450.0, inertia=0.1)  # En ağır bölge
        self.z4 = KilnZone(800.0,  inertia=0.3)

        self.o2, self.co2, self.p = 3.2, 18.0, -1.5

    def step(self, fuel_cmd, fan_cmd, feed_rate=120):
        # 1. Aktüatör durumları
        fuel, fan = self.act.step(fuel_cmd, fan_cmd)

        # 2. Isıl Kazanç ve Verimlilik (Profesyonel Bölümleme)
        LHV = 7500.0 # kcal/kg
        Q_chem = fuel * LHV
        
        T3K = self.z3.T + 273.15
        eta_t = 0.1 + 0.9 * (1 / (1 + np.exp(-0.012 * (T3K - 1473))))
        eta_o2 = 0.1 + 0.9 * (1 / (1 + np.exp(-7 * (self.o2 - 2.2))))
        eta_mix = np.exp(-((fan - 950) / 260) ** 2)

        # Toplam verim 0.11
        combustion_efficiency = 0.85 
        thermal_transfer_factor = 0.1294 
        system_efficiency = combustion_efficiency * thermal_transfer_factor
        
        burn_gen = Q_chem * eta_o2 * eta_t * eta_mix * system_efficiency

        # 3. Sıcaklık Güncellemeleri
        t1 = self.z1.update(0, self.z2.T, 0.006)
        t2 = self.z2.update(0, self.z3.T, 0.007)
        t3 = self.z3.update(burn_gen, self.z3.T, 0.009)
        t4 = self.z4.update(0, self.z3.T, 0.02)

        # 4. Gaz ve Basınç Dinamikleri
        air_flow = 2.1 / (1 + np.exp(-(fan - 950) / 240))
        eff = 1 / (1 + np.exp(-(self.o2 - 2.5)))
        o2_sink = 0.015 * burn_gen * eff
        self.o2 = np.clip(self.o2 + 0.1 * (0.21 * air_flow - o2_sink), 0.1, 5.0)
        self.co2 = np.clip(self.co2 + (1/30)*(0.04*burn_gen - self.co2*(fan/1000)), 2, 25)
        self.p += 0.1 * ((-3 + 1.7 * (air_flow - 1)) - self.p)

        # 5. Trend Akışkanlaştırma (Smooth Drift)
        d1 = self.noise.get_smooth_drift("T1", 0.01) * 30.0 
        d2 = self.noise.get_smooth_drift("T2", 0.01) * 25.0 
        d3 = self.noise.get_smooth_drift("T3", 0.012) * 20.0 
        d4 = self.noise.get_smooth_drift("T4", 0.008) * 15.0

        n1, n2, n3, n4 = [self.noise.process_noise(k, 0.03) for k in ["T1","T2","T3","T4"]]

        return {
            "T1": t1 + d1 + n1, "T2": t2 + d2 + n2,
            "T3": t3 + d3 + n3, "T4": t4 + d4 + n4,
            "o2": self.o2 + np.random.normal(0, 0.005),
            "co2": self.co2 + np.random.normal(0, 0.01),
            "pressure": self.p + np.random.normal(0, 0.002),
            "fuel": fuel, "fan": fan
        }
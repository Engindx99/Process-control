import numpy as np

# =========================================================
# 1. FİZİKSEL BİLEŞENLER (Temel Yapı Taşları)
# =========================================================
class KilnZone:
    def __init__(self, T0):
        self.T_mat, self.T_gas = T0, T0
        
    def update(self, heat_gen, conv_in, loss, inertia):
        # Gaz dinamiği: Gaz kütlesi az olduğu için hala hızlı (0.2 sabit)
        self.T_gas += 0.2 * (heat_gen + conv_in - self.T_gas)
        
        # Isı Kaybı: Çeperden dışarı sızan enerji
        cooling = loss * (self.T_mat - 25)
        
        # GÜÇLENDİRİLMİŞ ATALET (Inertia): 
        # Katsayı ne kadar küçükse, T_mat o kadar hantal hareket eder.
        self.T_mat += inertia * (self.T_gas - self.T_mat - cooling)
        return self.T_mat, self.T_gas

class ActuatorModel:
    def __init__(self):
        self.fuel, self.fan = 18.5, 950.0
        # Aktüatör tamponları (Gecikme süreleri)
        self.fuel_buf, self.fan_buf = [18.5]*60, [950.0]*30 
        
    def step(self, f_cmd, n_cmd):
        self.fuel_buf.append(np.clip(f_cmd, 0, 40))
        self.fan_buf.append(np.clip(n_cmd, 500, 1200))
        f, n = self.fuel_buf.pop(0), self.fan_buf.pop(0)
        # Valf motoru hızı (0.01 -> 0.005 yaparak daha da hantallaştırdık)
        self.fuel += 0.008 * (f - self.fuel)
        self.fan += 0.01 * (n - self.fan)
        return self.fuel, self.fan

# =========================================================
# 2. ANA SİSTEM (Digital Twin / Plant)
# =========================================================
class RotaryKilnPlant:
    def __init__(self):
        self.act = ActuatorModel()
        self.noise = NoiseModel()
        # Başlangıç bölgeleri
        self.z1 = KilnZone(850.0)  
        self.z2 = KilnZone(1100.0) 
        self.z3 = KilnZone(1450.0) 
        self.z4 = KilnZone(800.0)  
        self.o2, self.co2, self.p = 3.2, 18.0, -1.5

    def step(self, fuel_cmd, fan_cmd, feed_rate=120.0):
        fuel, fan = self.act.step(fuel_cmd, fan_cmd)

        # Yanma Verimi
        o2_target = np.clip(21 - fuel * 0.96, 1.5, 6.0)
        self.o2 += 0.05 * (o2_target - self.o2)
        eff = np.exp(-0.5 * ((self.o2 - 3.2) / 1.5) ** 2)
        burn_gen = fuel * eff * 86.9

        # --- GÜÇLENDİRİLMİŞ ATALETLİ BÖLGE GÜNCELLEMELERİ ---
        # Inertia değerleri bir öncekine göre %20-%50 oranında düşürüldü.
        t3, g3 = self.z3.update(burn_gen, 0, 0.008, 0.000012) # 0.0002 -> 0.00012
        t2, g2 = self.z2.update(0, g3 * 0.74, 0.006, 0.00004)  # 0.0006 -> 0.0004
        t1, g1 = self.z1.update(0, g2 * 0.78, 0.006, 0.00005)  # 0.0008 -> 0.0005
        t4, _  = self.z4.update(t3 * 0.75, 0, 0.1, 0.00008)   # 0.001 -> 0.0008

        # Kimyasal yükler
        self.z1.T_mat -= 0.00008 * feed_rate
        calc_load = np.clip(0.05 * np.exp(0.006 * (t2 - 1100)), 0, 5) if t2 > 900 else 0
        self.z2.T_mat -= 0.0008 * calc_load

        # Gazlar
        co2_gen = burn_gen * 0.02 + calc_load * 0.6
        self.co2 = np.clip(self.co2 + 0.01 * (co2_gen - self.co2 * (fan / 1000)), 2, 25)
        self.p += 0.1 * ((-2.0 * (fan / 1200)) + 0.18 - self.p)

        # --- GÖRSELLEŞTİRME KATMANI ---
        # Atalet arttığı için drift intensity'sini de biraz kıstık ki ani zıplamasın
        d1 = self.noise.get_smooth_drift("T1", 0.018) * 50.0 
        d2 = self.noise.get_smooth_drift("T2", 0.018) * 40.0
        d3 = self.noise.get_smooth_drift("T3", 0.02) * 35.0 
        d4 = self.noise.get_smooth_drift("T4", 0.012) * 30.0

        n1, n2, n3, n4 = [self.noise.process(k, 0.04) for k in ["T1","T2","T3","T4"]]

        return {
            "T1": t1 + d1 + n1, "T2": t2 + d2 + n2,
            "T3": t3 + d3 + n3, "T4": t4 + d4 + n4,
            "o2": self.o2 + (d3 * 0.004) + np.random.normal(0, 0.002),
            "co2": self.co2 + (d2 * 0.008) + np.random.normal(0, 0.005),
            "pressure": self.p + (d1 * 0.0004) + np.random.normal(0, 0.001),
            "fuel": fuel, "fan": fan
        }

# =========================================================
# 3. YARDIMCI ARAÇLAR (Gürültü ve Dalgalanma)
# =========================================================
class NoiseModel:
    def __init__(self):
        self.state = {k: 0.0 for k in ["T1","T2","T3","T4","o2","co2","p"]}
        self.drift_raw = {k: 0.0 for k in ["T1","T2","T3","T4"]}
        self.drift_smooth = {k: 0.0 for k in ["T1","T2","T3","T4"]}

    def process(self, key, sigma, alpha=0.95):
        eps = np.random.normal(0, sigma)
        self.state[key] = alpha * self.state[key] + eps
        return self.state[key]

    def get_smooth_drift(self, key, intensity=0.015):
        self.drift_raw[key] = (self.drift_raw[key] * 0.998) + np.random.normal(0, intensity)
        self.drift_smooth[key] = (self.drift_smooth[key] * 0.995) + (self.drift_raw[key] * 0.005)
        return self.drift_smooth[key]
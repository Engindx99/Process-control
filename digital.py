import numpy as np
import pandas as pd
import pickle
import os

class RotaryKilnDigitalTwin:
    def __init__(self):
        # --- Başlangıç Durumları ---
        self.temp = 950.0        # Celsius
        self.o2 = 4.5           # %
        self.fuel = 16.0        # Yakıt (kg/h)
        self.fan = 1000.0       # Fan Hızı (rpm)
        self.T_env = 25.0       # Ortam Sıcaklığı
        
        self.step_count = 0
        self.data = []
        
        # Gecikme (Dead Time): Yakıtın fırın içindeki yolculuğu (12 adım)
        self.fuel_history = [self.fuel] * 12 
        
        # --- Model Katsayıları ---
        self.thermal_mass = 0.07       # Isıl atalet
        self.heat_gain_factor = 7.2    # Yakıt enerji çarpanı
        self.conv_factor = 0.0004      # Taşınım katsayısı
        self.rad_factor = 5.67e-12     # Stefan-Boltzmann sabiti uyumlu
        
        self.phys_weight = 0.65        # Fiziksel denklemin ağırlığı
        self.emp_weight = 0.35         # Ampirik regresyonun ağırlığı

    def calculate_o2(self, fuel, fan):
        """Oksijen seviyesini yakıt ve hava girişine göre hesaplar."""
        base_o2 = 11.5 - 0.395 * fuel  # Ampirik temel
        physical_o2 = ((fan / 1000.0) * 15.5) - (fuel * 0.65) # Fiziksel denge
        return np.clip(0.70 * base_o2 + 0.30 * physical_o2, 2.0, 8.0)

    def calculate_efficiency(self, temp, o2):
        """Sıcaklık ve O2'ye bağlı yanma verimliliği."""
        temp_eff = np.clip(1.25 - 0.0005 * temp, 0.50, 0.99)
        o2_penalty = np.exp(-0.5 * ((o2 - 3.5) / 1.5) ** 2)
        return np.clip(temp_eff * (0.75 + 0.25 * o2_penalty), 0.45, 0.99)

    def step(self, fuel, fan):
        """Sistemi bir adım (1 dk) ilerletir."""
        self.fuel = np.clip(fuel, 12.0, 22.0)
        self.fan = np.clip(fan, 950.0, 1100.0)
        
        # Yakıt gecikmesini işlet
        self.fuel_history.append(self.fuel)
        delayed_fuel = self.fuel_history.pop(0)
        
        # O2 Hesapla
        target_o2 = self.calculate_o2(self.fuel, self.fan)
        self.o2 += 0.25 * (target_o2 - self.o2) # Sensör tepki hızı
        
        # Sıcaklık Hesabı (Fiziksel: Işınım + Taşınım)
        T_k = self.temp + 273.15
        Te_k = self.T_env + 273.15
        o2_eff = np.exp(-0.5 * ((self.o2 - 3.0) / 2.0) ** 2)
        
        heat_gain = delayed_fuel * self.heat_gain_factor * o2_eff
        heat_loss_conv = (0.01 + self.conv_factor * self.fan) * (self.temp - self.T_env)
        heat_loss_rad = (self.rad_factor * (T_k**4 - Te_k**4)) / 120 # Ölçeklendirilmiş radyasyon
        
        # Ampirik Hedef (Regresyon Modeli)
        target_temp = (410.0 * self.o2 - 172.0) - ((self.fan - 1000) / 1000 * 15)
        
        # Hibrit Diferansiyel Denklem
        dT = self.phys_weight * (heat_gain - heat_loss_conv - heat_loss_rad) + \
             self.emp_weight * (target_temp - self.temp)
        
        self.temp = np.clip(self.temp + self.thermal_mass * dT, 250, 1550)
        eff = self.calculate_efficiency(self.temp, self.o2)
        
        # Veri kaydı
        self.data.append({
            "adim": self.step_count,
            "fuel": round(self.fuel, 3),
            "fan": round(self.fan, 1),
            "sicaklik": round(self.temp, 2),
            "o2": round(self.o2, 3),
            "efficiency": round(eff, 4)
        })
        self.step_count += 1
        return self.temp, self.o2, eff

    def run_full_simulation(self, steps=5000, csv_file="firin_dataset_5k.csv", pkl_file="firin_model.pkl"):
        """5000 adımlık veri üretir ve tüm dosyaları kaydeder."""
        print(f"🔄 Simülasyon başladı: {steps} adım...")
        
        # Dinamik Girdi Üretimi (Random Walk)
        f_val, v_val = 16.0, 1000.0
        
        for _ in range(steps):
            f_val = np.clip(f_val + np.random.normal(0, 0.12), 13.0, 21.0)
            v_val = np.clip(v_val + np.random.normal(0, 2.5), 960.0, 1080.0)
            self.step(f_val, v_val)
            
        # --- KAYIT İŞLEMLERİ ---
        # 1. CSV Kaydı
        df = pd.DataFrame(self.data)
        df.to_csv(csv_file, index=False)
        
        # 2. Model (Nesne) Kaydı
        with open(pkl_file, 'wb') as f:
            pickle.dump(self, f)
            
        print("-" * 30)
        print(f"✅ BAŞARILI: {len(df)} satır veri üretildi.")
        print(f"📂 CSV Dosyası: {os.path.abspath(csv_file)}")
        print(f"📂 Model Dosyası: {os.path.abspath(pkl_file)}")
        print("-" * 30)

# --- ÇALIŞTIR ---
if __name__ == "__main__":
    # Dijital ikizi başlat
    kiln_twin = RotaryKilnDigitalTwin()
    
    # Simülasyonu çalıştır ve kaydet
    kiln_twin.run_full_simulation(steps=5000)
    
    # Kontrol amaçlı ilk 5 satırı yazdır
    data_check = pd.read_csv("firin_dataset_5k.csv")
    print("\nÜretilen Veriden Örnek (İlk 5 Satır):")
    print(data_check.head())
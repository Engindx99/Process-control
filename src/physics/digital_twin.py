import numpy as np

class RotaryKilnPhysics:
    def __init__(self):
        # Sabit Parametreler (Fabrika verileri gibi düşünebiliriz)
        self.mass = 5000.0         # Fırın içindeki malzemenin efektif kütlesi (kg)
        self.cp = 1.1              # Özgül ısı kapasitesi (kJ/kg.K)
        self.heat_transfer_coeff = 0.05 # Çevreye ısı kaybı katsayısı
        self.ambient_temp = 25.0    # Ortam sıcaklığı (°C)
        
        # Değişken Durumlar (States)
        self.current_temp = 1200.0  # Başlangıç iç sıcaklığı (°C)
        self.oxygen_level = 3.0     # O2 seviyesi (%)

    def calculate_dynamics(self, fuel_rate, material_feed, fan_speed):
        """
        Fırın dinamiklerini hesaplar (Euler integrasyonu ile basitleştirilmiş)
        fuel_rate: Yakıt giriş miktarı (kg/s)
        material_feed: Hammadde besleme hızı (kg/s)
        fan_speed: Fan devri (soğuma etkisi)
        """
        # 1. Isı Kazancı (Yakıt Alt Isıl Değeri ~ 30,000 kJ/kg kabul edilirse)
        q_in = fuel_rate * 30000 
        
        # 2. Isı Kayıpları (Radyasyon ve Konveksiyon basitleştirmesi)
        q_loss = self.heat_transfer_coeff * (self.current_temp - self.ambient_temp)
        
        # 3. Hammaddeye Aktarılan Isı
        q_material = material_feed * self.cp * (self.current_temp - 100) # 100 giriş temp varsayımı
        
        # 4. Sıcaklık Değişimi (dT/dt)
        # Basitleştirilmiş diferansiyel: dT = (Q_in - Q_loss - Q_material) / (m * Cp)
        dt = (q_in - q_loss - q_material) / (self.mass * self.cp)
        
        # Zaman adımı (Delta T = 1 saniye gibi düşünelim)
        self.current_temp += dt
        
        # Rastgele gürültü ekleyerek gerçekçiliği artıralım (Endüstriyel gürültü)
        self.current_temp += np.random.normal(0, 0.5)
        
        return self.current_temp

    def reset(self):
        """Sistemi başlangıç değerlerine döndürür"""
        self.current_temp = 1200.0
        return self.current_temp
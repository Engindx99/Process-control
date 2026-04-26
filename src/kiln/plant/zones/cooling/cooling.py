import numpy as np

class CoolerZone:
    def __init__(self, config=None):
        """
        Klinker Soğutucu (Grate Cooler) Dijital İkizi.
        Klinkerin 1450°C'den ~100°C'ye soğutulduğu bölge.
        """
        self.config = config if config else {}
        
        # Fiziksel Sabitler
        self.cp_clinker = 0.82       # kJ/kgK
        self.cp_air = 1.005          # kJ/kgK
        
        # Durum Değişkenleri
        self.T_clinker_out = 100.0    # Siloya giden klinker sıcaklığı
        self.T_secondary_air = 950.0  # Fırına giden ikincil hava
        self.T_tertiary_air = 850.0   # Kalsinatöre giden tersiyer hava

    def step(self, T_clinker_in, m_dot_clinker, m_dot_cooling_air, dt=1):
        """
        T_clinker_in: Burning Zone'dan çıkan klinker sıcaklığı (~1450°C)
        m_dot_clinker: Klinker debisi (tph)
        m_dot_cooling_air: Soğutma fanlarının toplam hava debisi (Nm3/h)
        """
        
        # 1. ISI TRANSFERİ (Klinker -> Hava)
        # Soğutucu verimi (Grate efficiency): Genelde %70-%80 arasıdır
        efficiency = self.config.get('cooler_efficiency', 0.72)
        
        # Klinkerden çekilen toplam enerji
        # Q = m * cp * deltaT
        q_total_available = m_dot_clinker * self.cp_clinker * (T_clinker_in - self.T_clinker_out) * efficiency * dt
        
        # 2. GERİ KAZANILAN ISININ DAĞILIMI
        # Geri kazanılan ısının bir kısmı fırına (Secondary), bir kısmı kalsinatöre (Tertiary) gider.
        # Geri kalanı ise bacadan (Exhaust air) atılır.
        
        # İkincil hava (Secondary Air) - Fırın için
        q_secondary = q_total_available * 0.55
        self.T_secondary_air = 25 + (q_secondary / (m_dot_cooling_air * 0.4 * self.cp_air + 1e-6))
        
        # Tersiyer hava (Tertiary Air) - Kalsinatör için
        q_tertiary = q_total_available * 0.35
        self.T_tertiary_air = 25 + (q_tertiary / (m_dot_cooling_air * 0.3 * self.cp_air + 1e-6))
        
        # 3. KLİNKER ÇIKIŞ SICAKLIĞI
        # Soğutma havası ne kadar fazlaysa, klinker o kadar soğur.
        cooling_power = m_dot_cooling_air * self.cp_air * efficiency
        self.T_clinker_out = T_clinker_in - (cooling_power / (m_dot_clinker * self.cp_clinker + 1e-6))
        
        # Fiziksel sınır: Klinker ortam sıcaklığının (25°C) altına düşemez
        self.T_clinker_out = max(25.0, self.T_clinker_out)

        return {
            "T_clinker_to_silo": self.T_clinker_out,
            "T_secondary_air": self.T_secondary_air,
            "T_tertiary_air": self.T_tertiary_air,
            "heat_recuperation_rate": efficiency * 100 # % verim
        }

# Hızlı Test
if __name__ == "__main__":
    cooler = CoolerZone()
    res = cooler.step(T_clinker_in=1400, m_dot_clinker=100, m_dot_cooling_air=250)
    print(f"Siloya giden klinker: {res['T_clinker_to_silo']:.2f} °C")
    print(f"Sekonder Hava (Fırına): {res['T_secondary_air']:.2f} °C")
    print(f"Tersiyer Hava (Kalsinatöre): {res['T_tertiary_air']:.2f} °C")

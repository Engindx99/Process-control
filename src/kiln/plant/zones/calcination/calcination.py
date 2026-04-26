import numpy as np

class CalcinationZone:
    def __init__(self, config):
        self.config = config
        
        # Fiziksel Sabitler
        self.heat_of_reaction = 1780   # kJ/kg (CaCO3 kalsinasyonu için gereken enerji)
        self.cp_material = 1.05        # kJ/kgK
        self.cp_gas = 1.2              # kJ/kgK
        self.lower_heating_value = 25000 # kJ/kg (Yakıt ısıl değeri)
        
        # Durum Değişkenleri (State Variables)
        self.calcination_degree = 0.1  # Başlangıç %10 (Preheater'dan gelen)
        self.T_mat = 850.0
        self.T_gas = 1000.0

    def step(self, T_mat_in, T_gas_in, T_tertiary_air, m_dot_meal, m_dot_fuel, m_dot_tertiary, dt=1):
        """
        T_mat_in: Preheater'dan gelen malzeme sıcaklığı
        T_gas_in: Fırın içinden gelen sıcak gaz (Burning Zone'dan)
        T_tertiary_air: Soğutucudan gelen sıcak hava
        """
        
        # 1. ENERJİ GİRİŞLERİ
        # Yakıtın yanmasından gelen ısı
        q_fuel = m_dot_fuel * self.lower_heating_value * dt
        
        # Tersiyer havadan gelen duyulur ısı
        q_tertiary = m_dot_tertiary * self.cp_gas * T_tertiary_air * dt
        
        # Fırından gelen gazın getirdiği ısı (Gaz debisi m_dot_tertiary ile orantılı varsayılmıştır)
        q_gas_from_kiln = m_dot_tertiary * self.cp_gas * T_gas_in * dt * 0.5 

        # 2. REAKSİYON KİNETİĞİ (Kalsinasyon)
        # Sıcaklık 800°C üzerine çıktıkça kalsinasyon hızı artar
        if self.T_mat > 800:
            reaction_speed = 0.05 * (self.T_mat - 800) / 100
        else:
            reaction_speed = 0.01
            
        # Kalsinasyon artışı hesapla (Hedef %95)
        calc_increment = (0.95 - self.calcination_degree) * reaction_speed * dt
        calc_increment = max(0, calc_increment) # Negatif artış olamaz
        self.calcination_degree += calc_increment
        
        # 3. ENDOTERMİK ENERJİ TÜKETİMİ
        # Reaksiyonun gerçekleşmesi için gereken enerji sistemden çekilir
        q_reaction = calc_increment * m_dot_meal * self.heat_of_reaction
        
        # 4. NET ENERJİ DENGESİ VE SICAKLIK GÜNCELLEME
        # Toplam giren enerji - Reaksiyonun tükettiği enerji
        total_energy_in = q_fuel + q_tertiary + q_gas_from_kiln
        net_energy_change = total_energy_in - q_reaction
        
        # Malzeme sıcaklığı güncelleme
        # T_new = T_old + Net_Q / (m * Cp)
        self.T_mat = T_mat_in + (net_energy_change / (m_dot_meal * self.cp_material + 1e-6))
        
        # Gaz çıkış sıcaklığı (Kalsinatör içi denge sıcaklığı tahmini)
        self.T_gas = (total_energy_in / (m_dot_tertiary * self.cp_gas + 1e-6)) * 0.9

        return {
            "T_mat_to_kiln": self.T_mat,
            "T_gas_to_preheater": self.T_gas,
            "calcination_degree": self.calcination_degree,
            "co2_produced": calc_increment * m_dot_meal * 0.44
        }
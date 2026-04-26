import numpy as np

class BurningZone:
    def __init__(self, config=None):
        """
        Gelişmiş Hava/Yanma Fiziği Entegre Edilmiş Döner Fırın Pişme Bölgesi.
        """
        self.config = config if config else {}
        
        # Fiziksel Sabitler
        self.cp_clinker = 0.85       
        self.cp_gas = 1.3            
        self.main_fuel_lhv = 7500    
        
        # Yanma Sabitleri
        # 1 kg kömür için gereken teorik hava miktarı (~10 kg hava)
        self.stochiometric_air_ratio = 10.5 
        
        # Durum Değişkenleri
        self.T_clinker = 1450.0      
        self.T_flame = 1800.0        
        self.T_shell = 320.0         

    def calculate_kiln_torque(self, m_dot_clinker, rpm):
        liquid_phase = max(0, (self.T_clinker - 1250) * 0.02) 
        base_load = m_dot_clinker * 0.5
        viscosity_effect = liquid_phase * 150
        torque = (base_load + viscosity_effect) * (1 + (rpm / 5.0))
        return torque

    def step(self, T_mat_in, m_dot_clinker, m_dot_main_fuel, m_dot_primary_air, rpm, dt=1):
        """
        Gelişmiş Yanma Fiziği Uygulanmış Adım Fonksiyonu
        """
        
        # --- 1. YANMA FİZİĞİ: HAVA/YAKIT DENGESİ (Lambda - λ) ---
        # Toplam hava (Basitleştirme: Primer hava üzerinden toplam yanma havası tahmini)
        # Gerçekte Sekonder hava daha büyüktür, burada primer havayı 'yanma havası' gibi modelliyoruz.
        theoretical_air_req = m_dot_main_fuel * self.stochiometric_air_ratio
        lambda_val = m_dot_primary_air / (theoretical_air_req + 1e-6)
        
        # Yanma Verimi Etkisi (Combustion Efficiency)
        # Lambda < 1.0 -> Eksik yanma (CO oluşumu, enerji kaybı)
        # Lambda > 1.1 -> Fazla hava (Alevin soğuması)
        if lambda_val < 1.0:
            # Yakıtın tamamı yanamaz (Eksik yanma kaybı)
            efficiency_penalty = 0.8 * lambda_val 
        elif lambda_val > 1.2:
            # Fazla hava alev sıcaklığını aşağı çeker (Soğutma etkisi)
            efficiency_penalty = 0.8 / (1 + (lambda_val - 1.2) * 2.0)
        else:
            # İdeal yanma aralığı (1.05 - 1.20)
            efficiency_penalty = 0.8
            
        q_combustion = m_dot_main_fuel * self.main_fuel_lhv * efficiency_penalty * dt
        
        # --- 2. ALEV KARAKTERİSTİĞİ (Flame Momentum) ---
        # Alevin momentumu (Hava/Yakıt hızı) ısı transfer katsayısını etkiler.
        # Primer hava hızı (momentum) arttıkça alev daha keskin ve radyasyon daha yoğun olur.
        flame_momentum = m_dot_primary_air / (m_dot_main_fuel + 1e-6)
        radyasyon_coeff = 0.00083 * (1 + (flame_momentum / 20.0)) # Momentum etkisi

        # --- 3. ISI TRANSFER HESAPLARI ---
        q_radiation = radyasyon_coeff * (self.T_flame**4 - self.T_clinker**4) * 1e-10 * dt
        q_exothermic = m_dot_clinker * 21.0 * dt
        q_loss_shell = 0.40 * (self.T_clinker - self.T_shell) * dt
        
        # --- 4. NET ENERJİ VE SICAKLIK GÜNCELLEME ---
        residence_time_factor = 2.5 / (rpm + 0.1)
        net_energy = q_radiation + q_exothermic - q_loss_shell
        
        # Klinker sıcaklığı
        self.T_clinker = T_mat_in + (net_energy / (m_dot_clinker * self.cp_clinker + 1e-6)) * residence_time_factor
        
        # Alev sıcaklığı: Yanma enerjisi / (Hava + Yakıt Kütlesi)
        # Lambda etkisi alev sıcaklığını doğal olarak etkiler
        total_mass = m_dot_main_fuel + m_dot_primary_air
        self.T_flame = 700 + (q_combustion / (total_mass * self.cp_gas + 1e-6))

        return {
            "T_clinker_out": self.T_clinker,
            "T_flame": self.T_flame,
            "lambda": lambda_val, # Yeni eklenen takip parametresi
            "kiln_torque": self.calculate_kiln_torque(m_dot_clinker, rpm),
            "status": "Normal" if self.T_clinker < 1550 else "Overheated"
        }
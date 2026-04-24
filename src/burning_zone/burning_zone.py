import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class RotaryKilnPlant:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.logger = logging.getLogger("KilnPlant")
        self.logger.setLevel(logging.INFO)

        self.reset()

    def reset(self):
        self.temp = 1450.0
        self.o2 = 2.2
        self.co2 = 14.0
        self.pressure = -2.0

        # Dijital İkiz Bileşenleri
        self.fuel_cmd = 18.5    # MPC'den gelen basamak sinyali (Command)
        self.fuel = 18.5        # Vanadan geçen gerçek akış (Continuous Flow)
        
        self.fan_cmd = 950.0    # Fan emir sinyali
        self.fan = 950.0        # Fan gerçek devri

        self.k_heat = 0.371
        self.tau_gas = 6.0
        self.tau_pressure = 18.0
        self.tau_co2 = 30.0

        self.dt = 1.0
        self.C_th = 1400 

        self.noise_o2 = 0.0
        self.noise_temp = 0.0
        self.noise_pressure = 0.0

        self.data = []
        self.step_count = 0

        return self.state()

    def state(self):
        # Kayıt setine hem emir (cmd) hem de gerçekleşen (fuel/fan) değerleri ekliyoruz
        return {
            "temp": self.temp,
            "o2": self.o2,
            "co2": self.co2,
            "pressure": self.pressure,
            "fuel_cmd": self.fuel_cmd, # Basamaklı sinyal katmanı
            "fuel": self.fuel,         # Sürekli akış katmanı (Flow Rate)
            "fan_cmd": self.fan_cmd,
            "fan": self.fan
        }

    def combustion_eff(self, o2):
        return 1.0 / (1.0 + np.exp(-(o2 - 2.5)))

    def air_flow(self):
        if self.fan < 50:
            return 0.0
        x = (self.fan - 950) / 300.0
        return 2.0 * (1 / (1 + np.exp(-x)))

    def heat_loss(self, T, fan):
        T_ref = 1450.0
        radiation = 0.33 * ((T / T_ref) ** 4.15 - (298 / T_ref) ** 4.15)
        convection = 0.0038 * (1 + 0.6 * np.tanh((fan - 900.0) / 2070.0)) * (T - 25)
        return radiation + convection

    def step(self, fuel_cmd, fan_cmd):
        # Emirleri sakla (Görselleştirmedeki 'step function' katmanı)
        self.fuel_cmd = fuel_cmd
        self.fan_cmd = fan_cmd

        # ACTUATOR DYNAMICS (Continuous Flow Rate Katmanı)
        # 0.1 ve 0.15 katsayıları 'tau' (zaman sabiti) görevini görür.
        # Bu işlem basamak sinyalini yumuşatarak fiziksel akışa (continuous) çevirir.
        self.fuel += 0.1 * (self.fuel_cmd - self.fuel)
        self.fan += 0.1 * (self.fan_cmd - self.fan)

        fuel = self.fuel # Hesaplamalarda artık 'gerçek akış' kullanılır
  
        # AIR FLOW
        air_flow = self.air_flow()
        O2_in = 0.21 * air_flow

        # PRESSURE
        resistance = 0.45 * self.pressure + 0.18 * (self.temp - 1400) / 300
        pressure_target = -3.0 + 1.8 * (air_flow - 1.0) - resistance
        self.pressure += (1.0 / self.tau_pressure) * (pressure_target - self.pressure)
        self.pressure += np.random.normal(0, 0.01)
        self.pressure = np.clip(self.pressure, -6.0, -1.0)
     
        # O2 DYNAMICS
        eff = self.combustion_eff(self.o2)
        o2_sink = 0.031 * fuel * eff
        mixing = 2 * (2.2 - self.o2) / self.tau_gas
        self.o2 += 0.1 * (O2_in - o2_sink + mixing)
        self.o2 += np.random.normal(0, 0.01)
        self.o2 = np.clip(self.o2, 0.1, 5.0)
        
        # COMBUSTION
        o2_gate = 1 / (1 + np.exp(-11 * (self.o2 - 1.0)))
        draft_effect = np.exp(-0.0000012 * (self.fan - 900) ** 2)
        combustion = fuel * o2_gate * draft_effect

        # --- CO2 DYNAMICS (Kalsinasyon Baskın & Pozitif Korelasyon) ---
        # 1. Yanma Kaynaklı Üretim Bileşenleri
        o2_consumed_local = 0.035 * combustion  # Yanma hızına bağlı O2 tüketimi
        co2_yield_local = 1.4 + 0.2 * o2_gate    # O2 varlığına bağlı verim
        
        # 2. Kalsinasyon Etkisi (Sıcaklık Arttıkça CO2'yi Zıplatan Terim)
        # 1400 derece üzerinde üstel artış sağlayarak sıcaklık grafiğiyle korelasyon kurar
        calc_effect = 0.6 * np.exp(0.018 * (self.temp - 1400)) if self.temp > 1300 else 0
        
        # 3. Toplam Üretim (Kalsinasyon Katsayısını Baskın Hale Getirdik)
        # Yanma etkisi (2.0) düşük, Sıcaklık etkisi (20.0) yüksek tutuldu
        co2_prod = (o2_consumed_local * co2_yield_local ) + (calc_effect * 4.0)
        
        # 4. Dinamik Güncelleme (Seyrelme ve Zaman Sabiti)
        dilution = self.fan / (self.fan + 1100.0) # Fan hızıyla gazın süpürülmesi
        self.co2 += (1.0 / self.tau_co2) * (
            co2_prod 
            - self.co2 * dilution 
            - 0.01 * (self.co2 - 12.0) # Geri çağırma etkisini azalttık
        )
        
        # 5. Gürültü ve Sınırlandırma
        
        self.noise_co2 = 0.9 * getattr(self, "noise_co2", 0.0) + np.random.normal(0, 0.005)
        self.co2 = np.clip(self.co2 + self.noise_co2, 5.0, 30.0)
        
        # HEAT GENERATION
        heat_gen = combustion * self.k_heat
        heat_gen *= (1 + 0.01 * np.sin(self.step_count / 400))

        # HEAT LOSS
        heat_loss = self.heat_loss(self.temp, self.fan)

        # TEMPERATURE DYNAMICS
        dT = (heat_gen - heat_loss) / self.C_th
        self.temp += self.dt * dT
        self.temp += np.random.normal(0, 0.03 * (self.temp / 1450.0))
        self.temp = np.clip(self.temp, 1200.0, 1650.0)

        # LOG
        self.step_count += 1
        record = self.state()
        record["step"] = self.step_count
        self.data.append(record)

        return record
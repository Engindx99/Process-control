import time
import numpy as np
from plant.zones.preheater.preheater import PreheaterTower
from plant.zones.calcination.calcination import CalcinationZone
from plant.zones.burning.burning import BurningZone
from plant.zones.cooling.cooling import CoolerZone

class KilnDigitalTwin:
    def __init__(self, config=None):
        self.config = config if config else {}
        
        # Fiziksel Bölgelerin Başlatılması
        self.preheater = PreheaterTower(self.config)
        self.calciner = CalcinationZone(self.config)
        self.burning_zone = BurningZone(self.config)
        self.cooler = CoolerZone(self.config)
        
        # Başlangıç Durumları
        self.t_tertiary_air = 800.0
        self.t_kiln_gas = 1100.0
        self.t_clinker_burning = 1450.0 

    def run_step(self, controls, dt=1):
        # 1. PREHEATER
        ph_out = self.preheater.compute_tower(
            T_raw_meal=25.0, 
            T_kiln_gas=self.t_kiln_gas, 
            m_meal_total=controls['feed_rate'], 
            m_gas_total=(controls['fan_speed'] * 1.8) + 10.0,
            dt=dt
        )
        
        # 2. CALCINATION (Tersiyer Hava Girişi)
        calc_out = self.calciner.step(
            T_mat_in=ph_out['feed_to_kiln_temp'],
            T_gas_in=self.t_kiln_gas,
            T_tertiary_air=self.t_tertiary_air,
            m_dot_meal=controls['feed_rate'],
            m_dot_fuel=controls['calciner_fuel'],
            m_dot_tertiary=controls['fan_speed'] * 0.8,
            dt=dt
        )
        
        # 3. BURNING ZONE
        bz_out = self.burning_zone.step(
            T_mat_in=calc_out['T_mat_to_kiln'],
            m_dot_clinker=controls['feed_rate'] * 0.65,
            m_dot_main_fuel=controls['main_fuel'],
            m_dot_primary_air=(controls['fan_speed'] * 0.1) + 2.0,
            rpm=controls['kiln_rpm'],
            dt=dt
        )
        
        # 4. COOLER (Fiziksel Motor)
        # Fan 0 olsa bile patlamayı önlemek için +5.0 hava akışı
        effective_cooling_air = controls['cooler_fans'] + 5.0
        cooler_out = self.cooler.step(
            T_clinker_in=self.t_clinker_burning,
            m_dot_clinker=controls['feed_rate'] * 0.65,
            m_dot_cooling_air=effective_cooling_air,
            dt=dt
        )
        
        # --- DİNAMİK GÜNCELLEMELER (HIZLI TEPKİ MODU) ---
        
        # Klinker Sıcaklığı: Fan arttıkça daha hızlı düşmesi için atalet %70'e çekildi
        yeni_t_clinker = bz_out['T_clinker_out']
        self.t_clinker_burning = (self.t_clinker_burning * 0.70) + (yeni_t_clinker * 0.30)
        
        # TERSİYER HAVA GÜNCELLEMESİ (ETKİ ARTIRILDI)
        raw_tertiary = cooler_out['T_tertiary_air']
        
        # Fan hassasiyeti artırıldı (Bölen 15.0). 
        # Fan 0 Hz -> çarpan ~1.0 | Fan 50 Hz -> çarpan ~0.23
        fan_sensitivity = 1.0 / (1.0 + (controls['cooler_fans'] / 15.0))
        
        # Baz sıcaklığı ve çarpanı yükselterek dinamik aralığı genişlettik
        yeni_t_tertiary = (raw_tertiary * 5.5) * fan_sensitivity + 350.0
        
        # Ataleti %70'e düşürdük; slider hareketini 3 saniye içinde tam görürsün
        self.t_tertiary_air = (self.t_tertiary_air * 0.70) + (yeni_t_tertiary * 0.30)
        
        # Fırın Gazı
        yeni_t_kiln_gas = bz_out['T_flame'] * 0.40 
        self.t_kiln_gas = (self.t_kiln_gas * 0.80) + (yeni_t_kiln_gas * 0.20)
        
        # Kalsinatör Görsel Limitör
        t_calc_display = calc_out['T_mat_to_kiln']
        if t_calc_display > 1050:
            t_calc_display = 900 + (t_calc_display * 0.05)

        return {
            "clinker_temp": self.t_clinker_burning,
            "T_calciner": t_calc_display,
            "T_preheater": ph_out['exit_gas_temp'],
            "T_tertiary": self.t_tertiary_air,
            "kiln_torque": bz_out['kiln_torque'],
            "calcination": calc_out['calcination_degree'],
            "lambda": bz_out.get('lambda', 1.1)
        }
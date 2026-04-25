import numpy as np

class RotaryKilnPlant:
    """
    Endüstriyel Döner Fırın Dijital İkiz Modeli.
    Çoklu yakıt girişi, ısıl atalet ve 20 dakikalık gecikmeli kalite tahmini içerir.
    """
    def __init__(self):
        # --- Bileşenlerin Başlatılması ---
        self.actuators = ActuatorSystem()
        self.signal_processor = SignalProcessor()
        
        # --- Isıl Bölgeler (Termal Düğümler) ---
        self.zones = {
            "preheat":    KilnZone(T0=850.0,  inertia=0.4),
            "calcination": KilnZone(T0=1100.0, inertia=0.2),
            "burning":     KilnZone(T0=1450.0, inertia=0.101),
            "cooling":     KilnZone(T0=800.0,  inertia=0.3)
        }

        # --- Gaz ve Basınç Durumları ---
        self.o2, self.co2, self.pressure = 3.2, 18.0, -1.5
        self._quality_buffer = [1.5] * 1200

    def step(self, fuel_main_cmd: float, fuel_alt_cmd: float, fan_cmd: float, feed_rate: float = 120.0) -> dict:
        """Sistemi bir adım ilerletir ve telemetri verilerini döner."""
        
        # 1. Aktüatör Dinamikleri
        total_fuel_cmd = fuel_main_cmd + fuel_alt_cmd
        actual_fuel, actual_fan = self.actuators.process(total_fuel_cmd, fan_cmd)

        # 2. Enerji ve LHV Hesabı
        lhv_mix = self._calculate_mixture_lhv(fuel_main_cmd, fuel_alt_cmd, total_fuel_cmd)
        q_chemical = actual_fuel * lhv_mix

        # 3. Yanma Verimliliği
        burn_energy = self._calculate_combustion_energy(q_chemical, actual_fan)

        # 4. Termal Yayılım (Sinyal İşleyici için T1-T4 anahtarlarıyla döner)
        t_zones = self._update_thermal_balance(burn_energy)

        # 5. Klinker Kalite Tahmini (fCaO)
        # t_zones içindeki "T3" anahtarını kullanıyoruz
        fcao_instant, fcao_delayed = self._process_quality_sensor(t_zones["T3"], feed_rate)

        # 6. Gaz Dinamikleri
        self._update_gas_dynamics(burn_energy, actual_fan)

        # 7. Sinyal İşleme (Gürültü ve Drift ekleme)
        return self.signal_processor.apply_telemetry(
            t_zones, fcao_instant, fcao_delayed, 
            self.o2, actual_fuel, lhv_mix, self.pressure
        )

    def _calculate_mixture_lhv(self, main, alt, total):
        if total <= 0: return 7500.0
        return ((main * 7500.0) + (alt * 5000.0)) / total

    def _calculate_combustion_energy(self, q_chem, fan):
        t3_k = self.zones["burning"].T + 273.15
        eta_t = 0.1 + 0.9 * (1 / (1 + np.exp(-0.012 * (t3_k - 1473))))
        eta_o2 = 0.1 + 0.9 * (1 / (1 + np.exp(-7 * (self.o2 - 2.2))))
        eta_mix = np.exp(-((fan - 950) / 260) ** 2)
        return q_chem * eta_o2 * eta_t * eta_mix * (0.85 * 0.1294)

    def _update_thermal_balance(self, burn_gen):
        t1 = self.zones["preheat"].update(0, self.zones["calcination"].T, 0.006)
        t2 = self.zones["calcination"].update(0, self.zones["burning"].T, 0.007)
        t3 = self.zones["burning"].update(burn_gen, self.zones["burning"].T, 0.009)
        t4 = self.zones["cooling"].update(0, self.zones["burning"].T, 0.02)
        
        # SignalProcessor ve fCaO hesabı "T1", "T2" vb. beklediği için bu şekilde dönüyoruz
        return {"T1": t1, "T2": t2, "T3": t3, "T4": t4}

    def _process_quality_sensor(self, t3, feed_rate):
        instant = np.clip(12.5 - 0.0075 * t3 + (feed_rate / 100), 0.5, 3.5)
        self._quality_buffer.append(instant)
        delayed = self._quality_buffer.pop(0)
        return instant, delayed

    def _update_gas_dynamics(self, burn_gen, fan):
        air_flow = 2.1 / (1 + np.exp(-(fan - 950) / 240))
        o2_sink = 0.015 * burn_gen * (1 / (1 + np.exp(-(self.o2 - 2.5))))
        self.o2 = np.clip(self.o2 + 0.1 * (0.21 * air_flow - o2_sink), 0.1, 5.0)
        self.pressure += 0.1 * ((-3 + 1.7 * (air_flow - 1)) - self.pressure)

# =========================================================
# YARDIMCI SİSTEMLER (Modüler Yapı)
# =========================================================

class KilnZone:
    def __init__(self, T0, inertia):
        self.T = T0
        self.inertia = inertia

    def update(self, Q_in, T_up, loss):
        conduction = 0.65 * (T_up - self.T)
        radiation = 3e-11 * ((self.T + 273)**4 - 298**4)
        dT = (Q_in + conduction - loss * (self.T - 25) - radiation) * (1.4e-5 * self.inertia)
        dT += 0.001 * np.sin(self.T / 30) 
        self.T += dT
        return self.T

class ActuatorSystem:
    def __init__(self):
        self.fuel, self.fan = 18.5, 950.0
        self._fuel_buffer = [18.5] * 80
        self._fan_buffer = [950.0] * 40

    def process(self, f_cmd, n_cmd):
        self._fuel_buffer.append(np.clip(f_cmd, 0, 40))
        self._fan_buffer.append(np.clip(n_cmd, 500, 1200))
        target_f, target_n = self._fuel_buffer.pop(0), self._fan_buffer.pop(0)
        self.fuel += 0.005 * (target_f - self.fuel)
        self.fan  += 0.01 * (target_n - self.fan)
        return self.fuel, self.fan

class SignalProcessor:
    def __init__(self):
        # Anahtarlar T1-T4 olarak standardize edildi
        self.drift_state = {k: [0.0, 0.0] for k in ["T1","T2","T3","T4"]}
        self.noise_state = {k: 0.0 for k in ["T1","T2","T3","T4"]}

    def apply_telemetry(self, t_zones, fcao_i, fcao_d, o2, fuel, lhv, press):
        results = {}
        scales = {"T1": 30.0, "T2": 25.0, "T3": 20.0, "T4": 15.0}
        
        for k in ["T1", "T2", "T3", "T4"]:
            temp = t_zones[k]
            # Drift hesabı
            self.drift_state[k][0] = (self.drift_state[k][0] * 0.999) + np.random.normal(0, 0.01)
            self.drift_state[k][1] = (self.drift_state[k][1] * 0.997) + (self.drift_state[k][0] * 0.003)
            # Noise hesabı
            eps = np.random.normal(0, 0.03)
            self.noise_state[k] = 0.95 * self.noise_state[k] + eps
            
            results[k] = temp + (self.drift_state[k][1] * scales[k]) + self.noise_state[k]

        results.update({
            "fCaO_instant": fcao_i, "fCaO_delayed": fcao_d,
            "o2": o2 + np.random.normal(0, 0.005),
            "pressure": press + np.random.normal(0, 0.002),
            "fuel_total": fuel, "LHV_mix": lhv
        })
        return results
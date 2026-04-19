import numpy as np
import pandas as pd
import logging

class RotaryKilnDigitalTwin:
    def __init__(self, config=None, log_level=logging.INFO, seed=None):
        # ---------------- SEEDING ----------------
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

        # ---------------- CONFIG MANAGEMENT ----------------
        # Eğer dışarıdan config gelmezse veya eksik gelirse sistemin çökmemesi için
        # güvenli bir sözlük yapısı kuruyoruz.
        self.config = config if config else {}
        
        # ---------------- LOGGING ----------------
        self.logger = logging.getLogger("RotaryKilnDT")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        self.reset()

    def reset(self):
        """Sistemi 1300°C başlangıç noktasına resetler."""
        self.temp = 1450.0
        self.o2 = 3.2 
        self.fuel = 18.5
        self.fan = 950.0
        self.T_env = 25.0
        self.step_count = 0
        self.data = []
        
        # KeyError: 'physics' hatasını önlemek için güvenli okuma:
        # Config'de yoksa varsayılan olarak 12 adımlık gecikme kullanır.
        physics_cfg = self.config.get("physics", {})
        d_steps = physics_cfg.get("delayed_steps", 12)
        
        # Yakıt hattındaki gecikme kuyruğu
        self.fuel_history = [self.fuel] * d_steps
        return self._get_obs()

    def _get_obs(self):
        return {"temp": self.temp, "o2": self.o2, "fuel": self.fuel, "fan": self.fan}

    def combustion_eff(self, o2):
        """Oksijene bağlı yanma verimliliği (Çan eğrisi)."""
        return 0.6 + 0.4 * np.exp(-0.5 * ((o2 - 3.2) / 1.4) ** 2)

    def get_staircase_target(self, current_minute):
        """Merdiven stratejisi: Her 60 dk'da bir +20°C artış, final 1450°C."""
        if current_minute < 60: return 1300
        elif current_minute < 120: return 1320
        elif current_minute < 180: return 1340
        elif current_minute < 240: return 1360
        else: return 1450

    def step(self, target_fuel, target_fan):
        """Fizik motorunun tek bir adımı."""
        try:
            # Limitleri config'den (limits veya plant başlığından) güvenli çek:
            limits = self.config.get("limits", self.config.get("plant", {}))
            f_min = limits.get("fuel_min", 12.0)
            f_max = limits.get("fuel_max", 28.0)
            v_min = limits.get("fan_min", 850.0)
            v_max = limits.get("fan_max", 1100.0)

            target_fuel = np.clip(target_fuel, f_min, f_max)
            target_fan = np.clip(target_fan, v_min, v_max)

            # Fiziksel Parametreler (Config'de yoksa standart fırın fiziği kullanılır)
            phys = self.config.get("physics", {})
            thermal_mass = phys.get("thermal_mass", 1050.0)
            heat_gain_f = phys.get("heat_gain_factor", 19.87)
            conv_f = phys.get("conv_factor", 0.00025)
            alpha = phys.get("fan_inertia", 0.15)

            # 1. FAN ATALETİ
            self.fan += alpha * (target_fan - self.fan)

            # 2. YAKIT GECİKMESİ
            self.fuel = target_fuel
            self.fuel_history.append(self.fuel)
            delayed_fuel = self.fuel_history.pop(0)

            # 3. O2 DİNAMİĞİ
            fuel_effect = 1.2 * np.tanh(0.18 * (self.fuel - 16))
            fan_effect = 1.5 * np.tanh((self.fan - 1000) / 120)
            o2_target = np.clip(3.4 + fan_effect - fuel_effect, 2.0, 5.0)
            self.o2 += 0.12 * (o2_target - self.o2)

            # 4. ISI DENGESİ
            eff = self.combustion_eff(self.o2)
            heat_gain = delayed_fuel * heat_gain_f * eff
            heat_loss = (0.004 + conv_f * self.fan) * (self.temp - self.T_env)
            
            delta_temp = (heat_gain - heat_loss) / thermal_mass
            delta_temp = np.clip(delta_temp, -5.0, 5.0)
            
            self.temp += delta_temp + np.random.normal(0, 0.14)

            current_target = self.get_staircase_target(self.step_count * (self.config.get("system", {}).get("step_duration_sec", 5) / 60))

            record = {
                "step": self.step_count,
                "target_Temp": float(current_target),
                "temperature": float(self.temp),
                "fuel": float(self.fuel),
                "fan": float(self.fan),
                "o2": float(self.o2),
                "efficiency": float(eff),
                "error": float(self.temp - current_target)
            }
            self.data.append(record)
            self.step_count += 1
            return record

        except Exception as e:
            self.logger.error(f"Step Error: {e}")
            raise

    def run_staircase_simulation(self, total_steps=360):
        """
        Merdiven senaryosunu baştan sona koşturur.
        Bu metod, MPC veya RL entegrasyonu için şablon niteliğindedir.
        """
        self.logger.info(f"Staircase simulation started for {total_steps} minutes.")
        self.reset()
        
        for i in range(total_steps):
            # Burada normalde MPC veya RL karar verir. 
            # Test amaçlı basit bir takip mantığı (placeholder):
            target_t = self.get_staircase_target(i)
            
            # Basit bir P-Kontrolcü mantığı ile test (MPC yerine):
            fuel_cmd = 16.0 + (target_t - self.temp) * 0.5
            fan_cmd = 950.0 + (self.o2 - 3.2) * 100
            
            self.step(fuel_cmd, fan_cmd)

        return pd.DataFrame(self.data)
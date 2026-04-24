import optuna
import yaml
import numpy as np
import copy
import logging
from src.burning_zone.burning_zone import RotaryKilnPlant
from src.mpc.mpc import MPC

# --- KALMAN FILTRESI (Gürültü Süzücü) ---
class KalmanFilter:
    def __init__(self, config):
        # Başlangıç tahminleri [O2, Basınç]
        self.x = np.array([2.5, -2.0], dtype=float) 
        n = len(self.x)
        self.P = np.eye(n) * 0.1
        
        # Süreç ve Ölçüm Gürültüsü (Config'den dinamik okuma)
        q_o2 = config.get('kalman', {}).get('q_o2', 0.0001)
        q_press = config.get('kalman', {}).get('q_press', 0.001)
        self.Q = np.diag([q_o2, q_press])
        
        r_val = config.get('kalman', {}).get('measurement_noise', 0.08)
        self.R = np.eye(n) * r_val
        
        self.F = np.eye(n)
        self.H = np.eye(n)

    def filter(self, o2_raw, pressure_raw):
        z = np.array([o2_raw, pressure_raw], dtype=float)
        # Tahmin (Predict)
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # Güncelleme (Update)
        y = z - (self.H @ x_pred)
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ P_pred
        return self.x[0], self.x[1]

# --- OPTUNA CALISMASI ---
with open("config.yaml", "r", encoding="utf-8") as f:
    BASE_CONFIG = yaml.safe_load(f)

def objective(trial):
    current_cfg = copy.deepcopy(BASE_CONFIG)
    
    # --- v10.16: Ultra-Smooth Search Space ---
    # Fanın uyumasını engellemek için ağırlığını biraz daha esnek bıraktık
    current_cfg['mpc'].update({
        'fuel_min': 10.0,
        'fuel_max': 25.0,
        'fan_min': 0.0,
        'fan_max': 1200.0,
        
        
        'max_delta_fuel': trial.suggest_float('max_delta_fuel', 0.05, 0.1), 
        'max_delta_fan': trial.suggest_float('max_delta_fan', 20.0, 100.0),  
        
        # Ağırlıklar: Sıcaklık baskısını azaltıp vana korumayı (fuel_chg) dengeledik
        'weight_temp': trial.suggest_float('weight_temp', 100000.0, 150000.0),
        'weight_o2': trial.suggest_float('weight_o2', 2000.0, 12000.0), 
        
       
        'weight_fuel_chg': trial.suggest_float('weight_fuel_chg', 20000, 100000.0),
        'weight_fan_chg': trial.suggest_float('weight_fan_chg', 5000.0, 15000.0),
        
        'prediction_horizon': 130,
        'control_horizon': 20
    })

    # Bileşenleri Başlat
    plant = RotaryKilnPlant(seed=42)
    obs = plant.reset()
    mpc = MPC(current_cfg) 
    kf = KalmanFilter(current_cfg)

    TOTAL_SECONDS = 21600 # 6 Saatlik test
    control_step = 10
    step_scores = []
    fuel_history = [obs['fuel']] * 10 
    fuel_cmd, fan_cmd = obs['fuel'], obs['fan']

    for i in range(TOTAL_SECONDS):
        # 1. Kalman Filtreleme
        filt_o2, filt_press = kf.filter(obs['o2'], obs['pressure'])

        if i % control_step == 0:
            try:
                # 2. MPC Aksiyonu
                fuel_cmd, fan_cmd = mpc.get_action(
                    obs['temp'], 
                    filt_o2, 
                    filt_press, 
                    fuel_history
                )
            except Exception:
                return 200000.0 # Çözücü çökerse yüksek ceza
        
        # 3. Fiziksel Adım
        obs = plant.step(fuel_cmd, fan_cmd)
        
        # Gecikme (lag) için geçmişi güncelle
        fuel_history.append(obs['fuel'])
        fuel_history.pop(0)
        
        # 4. Skorlama
        if i % control_step == 0:
            t_err = abs(obs['temp'] - 1450.0)
            o2_err = abs(obs['o2'] - 2.2)
            
            # Zaman ağırlığı: Son saatlerdeki hataya 10 kat daha fazla önem ver
            time_weight = 1.0 + (i / TOTAL_SECONDS) * 9.0 
            combined_err = t_err + (o2_err * 4.0) # O2 dengesi için çarpan 4.0
            step_scores.append(combined_err * time_weight)
            
            # Güvenlik Kesicisi: Çok büyük sapmalarda trial'ı iptal et
            if i > 3000:
                if t_err > 40 or o2_err > 2.0:
                    return 150000.0

    return np.mean(step_scores)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="kiln_v10_16_ultra_smooth", 
        direction='minimize', 
        storage="sqlite:///fast_kiln.db", 
        load_if_exists=True
    )
    
    print("🚀 Ultra-Smooth Optimizasyon Başlatıldı (n_trials=20)...")
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*40)
    print("🏆 EN İYİ SONUÇLAR (v10.16)")
    print(f"Skor: {study.best_value:.4f}")
    print("Parametreler:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*40)
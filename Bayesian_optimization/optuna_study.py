import optuna
import yaml
import numpy as np
import copy
from src.burning_zone.burning_zone import RotaryKilnPlant
from src.mpc.mpc import MPC


class KalmanFilter:
    def __init__(self, config):
        # Başlangıç tahminleri [O2, Basınç]
        self.x = np.array([2.5, -2.0], dtype=float) 
        n = len(self.x)
        self.P = np.eye(n) * 0.1
        
        # Süreç ve Ölçüm Gürültüsü
        q_o2 = config.get('kalman', {}).get('q_o2', 0.0001)
        q_press = config.get('kalman', {}).get('q_press', 0.001)
        self.Q = np.diag([q_o2, q_press])
        
        r_val = config.get('kalman', {}).get('measurement_noise', 0.08)
        self.R = np.eye(n) * r_val
        
        self.F = np.eye(n)
        self.H = np.eye(n)

    def filter(self, o2_raw, pressure_raw):
        z = np.array([o2_raw, pressure_raw], dtype=float)
        # Tahmin
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # Güncelleme
        y = z - (self.H @ x_pred)
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ P_pred
        return self.x[0], self.x[1]

# --- OPTUNA STUDY ---
with open("config.yaml", "r", encoding="utf-8") as f:
    BASE_CONFIG = yaml.safe_load(f)

def objective(trial):
    current_cfg = copy.deepcopy(BASE_CONFIG)
    
    # v10.10: Golden Lockdown + Kalman Parameters
   # v10.11: Genişletilmiş Sınırlar - Kalman Sonrası Yeni Dengeleri Bulma
    current_cfg['mpc'].update({
        'fuel_max': 30.0,
        'fan_max': 1200.0,
        
        # Hız Limitleri (Biraz daha esneklik tanıdık: 1.4 - 1.7)
        # Kalman gürültüyü süzdüğü için vana hızını artırmak artık tehlikeli değil
        'max_delta_fuel': trial.suggest_float('max_delta_fuel', 1.1, 1.5), 
        'max_delta_fan': trial.suggest_float('max_delta_fan', 90.0, 120.0),  
        
        # Ağırlıklar (130k - 150k ve 8k - 10k)
        # O2 ağırlığını 10k'ya kadar yolu var, bakalım Kalman sonrası ne diyecek
        'weight_temp': trial.suggest_float('weight_temp', 125000.0, 150000.0),
        'weight_o2': trial.suggest_float('weight_o2', 8500.0, 11000.0), 
        
        'weight_fuel_chg': trial.suggest_float('weight_fuel_chg', 50000.0, 100000.0),
        'weight_fan_chg': trial.suggest_float('weight_fan_chg', 8000.0, 15000.0),
        
        'prediction_horizon': 130,
        'control_horizon': 20
    })

    # Kurulum
    plant = RotaryKilnPlant(seed=42)
    obs = plant.reset()
    mpc = MPC(current_cfg) 
    
    # Her trial başında filtreyi sıfırla
    kf = KalmanFilter(current_cfg)

    TOTAL_SECONDS = 21600 
    control_step = 10
    step_scores = []
    fuel_history = [obs['fuel']] * 10 
    fuel_cmd, fan_cmd = obs['fuel'], obs['fan']

    for i in range(TOTAL_SECONDS):
        # 1. Ham veriyi Kalman'dan geçir
        filt_o2, filt_press = kf.filter(obs['o2'], obs['pressure'])

        if i % control_step == 0:
            try:
                # 2. MPC'ye temizlenmiş veriyi ver
                fuel_cmd, fan_cmd = mpc.get_action(
                    obs['temp'], 
                    filt_o2, 
                    filt_press, 
                    fuel_history
                )
            except Exception:
                return 150000.0 
        
        # 3. Plant her zaman gerçek dünyada olduğu gibi ham komutla sürülür
        obs = plant.step(fuel_cmd, fan_cmd)
        
        fuel_history.append(obs['fuel'])
        fuel_history.pop(0)
        
        # 4. Skorlama (Gerçek performans ölçümü için ham 'obs' kullanılır)
        if i % control_step == 0:
            t_err = abs(obs['temp'] - 1450.0)
            o2_err = abs(obs['o2'] - 2.2)
            time_weight = 1.0 + (i / TOTAL_SECONDS) * 9.0 
            combined_err = t_err + (o2_err * 5.0)
            step_scores.append(combined_err * time_weight)
            
            if i > 2000:
                if t_err > 35 or o2_err > 1.5:
                    return 120000.0

    return np.mean(step_scores)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="kiln_v10_10_kalman_final", 
        direction='minimize', 
        storage="sqlite:///fast_kiln.db", 
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)
    
    print("\n--- Kalman Entegre Optimizasyon Tamamlandı ---")
    print(f"En iyi Skor: {study.best_value}")
    print("Parametreler:", study.best_params)
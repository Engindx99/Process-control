import optuna
import yaml
import numpy as np
import copy
from src.dt.dt import RotaryKilnPlant
from src.mpc.mpc import MPC

# 1. Config'i bir kez yükle
with open("config.yaml", "r", encoding="utf-8") as f:
    BASE_CONFIG = yaml.safe_load(f)

def objective(trial):
    current_cfg = copy.deepcopy(BASE_CONFIG)
    
    # v9.9: "Thermal Breakthrough" - Atalet Kıran Agresif Ayarlar
    current_cfg['mpc'].update({
        'fuel_max': 120.0,
        'fan_max': 3500.0,
        
        # --- HIZ LİMİTLERİNDE RADİKAL ARTIŞ ---
        # 0.11 yetmiyordu, 0.3 ile 0.8 arasını deneyerek fırını hızlandırıyoruz
        'max_delta_fuel': trial.suggest_float('max_delta_fuel', 0.3, 0.8), 
        # Fanın yakıta yetişmesi için limitini esnetiyoruz
        'max_delta_fan': trial.suggest_float('max_delta_fan', 40.0, 80.0),  
        
        # --- AĞIRLIKLARDA NANO-ODAKLANMA ---
        # Önceki en iyi değerlerin (3M ve 394) etrafına hapsedildi
        'weight_temp': trial.suggest_float('weight_temp', 2998000.0, 3002000.0),
        'weight_o2': trial.suggest_float('weight_o2', 393.0, 395.0),
        'weight_fuel_chg': trial.suggest_float('weight_fuel_chg', 83000.0, 83200.0),
        'weight_fan_chg': trial.suggest_float('weight_fan_chg', 9700.0, 9750.0),
        
        'prediction_horizon': 200,
        'control_horizon': 20
    })

    plant = RotaryKilnPlant(seed=42)
    obs = plant.reset()
    mpc = MPC(current_cfg) 
    
    TOTAL_SECONDS = 21600 # 6 Saatlik Maraton
    control_step = 10
    errors = []
    fuel_cmd, fan_cmd = 23.0, 950.0

    for i in range(TOTAL_SECONDS):
        if i % control_step == 0:
            try:
                fuel_cmd, fan_cmd = mpc.get_action(obs['temp'], obs['o2'], plant.history_fuel)
            except: 
                return 100000.0 # Solver çökmesi durumunda yüksek ceza
        
        obs = plant.step(fuel_cmd, fan_cmd)
        
        if i % control_step == 0:
            err = abs(obs['temp'] - 1450.0)
            # Zaman ilerledikçe hatayı daha çok cezalandır (Stabilite testi)
            time_weight = 1.0 + (i / TOTAL_SECONDS) * 9.0 
            errors.append(err * time_weight)
            
            # Aşırı sapmada trial'ı erken durdur (Zaman kazan)
            if err > 70: return 90000.0

    return np.mean(errors)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="kiln_v9_9_breakthrough", 
        direction='minimize', 
        storage="sqlite:///fast_kiln.db", 
        load_if_exists=True
    )
    # Aralıklar dar olduğu için 20 trial en iyi noktayı bulmaya yetecektir
    study.optimize(objective, n_trials=20)
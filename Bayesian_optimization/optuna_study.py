import optuna
import yaml
import numpy as np
import copy
from src.burning_zone.burning_zone import RotaryKilnPlant
from src.mpc.mpc import MPC

# 1. Config'i bir kez yükle
with open("config.yaml", "r", encoding="utf-8") as f:
    BASE_CONFIG = yaml.safe_load(f)

def objective(trial):
    current_cfg = copy.deepcopy(BASE_CONFIG)
    
    # v10.6: "The Golden Ratio" - Kusursuz Denge
    current_cfg['mpc'].update({
        'fuel_max': 25.0,
        'fan_max': 1200.0,
        
        # --- HIZ LİMİTLERİ: 1.20 sınırını 1.50'ye, fanı 200'e çekiyoruz ---
        'max_delta_fuel': trial.suggest_float('max_delta_fuel', 1.00, 1.50), 
        'max_delta_fan': trial.suggest_float('max_delta_fan', 100.0, 200.0),  
        
        # --- AĞIRLIKLAR: 177k ve 3473 merkezli dar ama derin bir tarama ---
        'weight_temp': trial.suggest_float('weight_temp', 150000.0, 250000.0),
        'weight_o2': trial.suggest_float('weight_o2', 3000.0, 5000.0),
        
        # Değişim cezaları: 20k ve 1.5k civarı çok stabil çalıştı
        'weight_fuel_chg': trial.suggest_float('weight_fuel_chg', 18000.0, 35000.0),
        'weight_fan_chg': trial.suggest_float('weight_fan_chg', 1000.0, 4000.0),
        
        'prediction_horizon': 100,
        'control_horizon': 20
    })

    # ... (Kurulumlar aynı)
    # Ceza eşiğini 30 dereceye indirebilirsin. Bu skora ulaşan bir model 30 dereceden fazla sapmamalı.
    

    # ... (Kurulumlar aynı)
    # Erken durdurma eşiğini 40 derecede tutabilirsin, artık hata marjımız çok düşük.

    # ... (Geri kalan kurulumlar aynı)

    plant = RotaryKilnPlant(seed=42)
    obs = plant.reset()
    mpc = MPC(current_cfg) 
    
    TOTAL_SECONDS = 21600 
    control_step = 10
    errors = []
    fuel_cmd, fan_cmd = 18.0, 900.0

    for i in range(TOTAL_SECONDS):
        if i % control_step == 0:
            try:
                fuel_cmd, fan_cmd = mpc.get_action(obs['temp'], obs['o2'], plant.history_fuel)
            except: 
                return 100000.0 
        
        obs = plant.step(fuel_cmd, fan_cmd)
        
        if i % control_step == 0:
            err = abs(obs['temp'] - 1450.0)
            time_weight = 1.0 + (i / TOTAL_SECONDS) * 9.0 
            errors.append(err * time_weight)
            
            # Aralığı genişlettiğimiz için başta hata payı bırakalım
            if err > 30: return 90000.0

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
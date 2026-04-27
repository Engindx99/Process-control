import optuna
import yaml
import os
import sys
import json
from datetime import datetime

# --- AYARLAR ---
optuna.logging.set_verbosity(optuna.logging.INFO)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from main import run_simulation 

def objective(trial):
    root_config_path = os.path.join(BASE_DIR, "config.yaml")
    temp_configs_dir = os.path.join(BASE_DIR, "temp_configs")
    os.makedirs(temp_configs_dir, exist_ok=True)
    
    # --- OPTUNA ARAMA UZAYI (V4 - AGRESİF PİŞİRME) ---
    params = {
        'weight_calc': trial.suggest_float('weight_calc', 2000, 5000), # Kalsinasyon artık kolay, ağırlığı azalttık
        'weight_c2s': trial.suggest_float('weight_c2s', 50000, 100000),
        # Alev gücünü (C3S baskısı) zirveye çıkarıyoruz
        'weight_quality': trial.suggest_float('weight_quality', 10000000, 100000000), 
        'weight_smooth': trial.suggest_float('weight_smooth', 1000, 4000),
        # Senin config'deki 0.0001'i ezip 10 kat yavaşlatıyoruz
        'velocity': trial.suggest_float('velocity', 0.5e-05, 1.2e-05) 
    }

    try:
        with open(root_config_path, "r") as f:
            cfg_data = yaml.safe_load(f)
        
        # Optuna parametrelerini config'e bas
        cfg_data.update(params)
        
        # --- SİMÜLASYON HIZLANDIRMA VE GARANTİLEME ---
        if 'simulation' in cfg_data:
            # Hız düştüğü için adım sayısını artırıyoruz (Malzeme fırın sonuna ulaşmalı)
            cfg_data['simulation']['steps'] = 60000 
            cfg_data['simulation']['control_interval'] = 250 # CPU yükünü azaltır
            cfg_data['simulation']['save_interval'] = 100000 # Terminal kirliliğini önler

        temp_cfg_path = os.path.join(temp_configs_dir, f"config_trial_{trial.number}.yaml")
        with open(temp_cfg_path, "w") as f:
            yaml.dump(cfg_data, f)

        # Simülasyonu koştur
        final_c3s, final_c2s, stability_penalty = run_simulation(
            temp_cfg_path, 
            show_plots=False, 
            verbose=False
        )
        
        # --- SKOR HESAPLAMA (MAXIMIZE) ---
        c3s_pct = final_c3s * 100
        c2s_pct = final_c2s * 100
        
        # Skor formülü: Hedefimiz doğrudan %60+ C3S (Skor 60+)
        fitness = (c3s_pct * 1.5) + (c2s_pct * 0.1) - (stability_penalty * 0.005)
        
        # Başarısız (soğuk) fırınları doğrudan 0 ile cezalandır
        return fitness if c3s_pct > 0.01 else 0.0

    except Exception as e:
        print(f"Hata oluştu: {e}")
        return 0.0

if __name__ == "__main__":
    # Temiz bir başlangıç için DB adını değiştirdik
    db_path = os.path.join(BASE_DIR, "kiln_final_push.db")
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="kiln_ultimate_optimization",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )
    
    print("\n" + "*"*50)
    print("ULTIMATE PUSH MODU BAŞLATILDI")
    print("Hedef: %65 C3S | Velocity: 1e-5 seviyesi")
    print("*"*50 + "\n")
    
    study.optimize(objective, n_trials=40)

    # En iyi sonuçları kaydet
    best_file = os.path.join(BASE_DIR, "best_params_ultimate.json")
    if len(study.trials) > 0:
        with open(best_file, "w") as f:
            json.dump({
                "best_score": study.best_value,
                "best_params": study.best_params,
                "trial": study.best_trial.number
            }, f, indent=4)
        print(f"\n[BAŞARI] En iyi skor: {study.best_value:.2f}")
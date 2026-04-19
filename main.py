import os
import logging
import pandas as pd
import yaml
import time
from src.dt.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC
from src.rl.rl import train_rl

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_pure_mpc_benchmark(config):
    output_path = "data/pure_mpc_results.csv"
    
    if os.path.exists(output_path):
        logger.info(f"--- [KONTROL] {output_path} bulundu. MPC atlanıyor. ---")
        return

    logger.info("--- [MPC] Simülasyon Başlatılıyor... ---")
    plant = RotaryKilnDigitalTwin(config=config)
    plant.reset() 
    mpc = MPC(config)
    
    total_steps = int(24 * 3600 / config["system"]["step_duration_sec"]) 

    for i in range(total_steps):
        fuel_cmd, fan_cmd = mpc.optimize(plant, 1450.0)
        result = plant.step(fuel_cmd, fan_cmd)
        
        if i % 1440 == 0:
            temp = result.get('temperature') or result.get('Temperature', 0)
            logger.info(f"İlerleme: %{100*i/total_steps:.1f} | Temp: {temp:.2f}°C")

    df = pd.DataFrame(plant.data)
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(output_path, index=False)
    logger.info(f"--- [BAŞARI] Benchmark kaydedildi. ---")

def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} dosyası bulunamadı!")
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Windows/Linux Multiprocessing güvenliği için zorunlu
    try:
        # 1. Klasörler
        for folder in ["data", "models", "logs"]:
            if not os.path.exists(folder): os.makedirs(folder)

        # 2. Config
        cfg = load_config()
        logger.info("Config başarıyla yüklendi.")

        # 3. MPC (Dosya kontrolü fonksiyon içinde yapılıyor)
        generate_pure_mpc_benchmark(cfg)

        # 4. RL EĞİTİMİ
        logger.info("--- [RL] Stable-Baselines3 Başlatılıyor (CPU'lar hazırlanıyor...) ---")
        
        # CPU'ların ayağa kalkması için kısa bir bekleme (Bazen crash'i önler)
        time.sleep(2) 
        
        train_rl(cfg)

    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.error(f"KRİTİK HATA: {e}", exc_info=True)
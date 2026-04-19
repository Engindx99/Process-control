import os
import logging
import pandas as pd
import yaml
from src.dt.dt import RotaryKilnDigitalTwin
from src.mpc.mpc import MPC
from src.rl.rl import train_rl

# ---------------- LOGGING AYARI ----------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def generate_pure_mpc_benchmark(config):
    """
    Sabit 1450°C Benchmark Senaryosu.
    Fırın 1450'de başlar, hedef hep 1450'dir.
    """
    logger.info("--- 24 Saatlik SABİT 1450°C MPC Analizi Başlatılıyor ---")
    
    # 1. Dijital İkiz Kurulumu
    # NOT: dt.py içindeki reset() metodunda temp=1450 yaptığından emin ol!
    plant = RotaryKilnDigitalTwin(config=config, seed=42)
    
    # 2. MPC Kontrolcü Kurulumu
    mpc = MPC(config)
    
    # Zaman parametreleri
    step_sec = config["system"]["step_duration_sec"]
    total_steps = int(24 * 3600 / step_sec) # 24 saat
    
    logger.info(f"Simülasyon toplam {total_steps} adım sürecek.")

    for i in range(total_steps):
        # HEDEF SABİT 1450
        current_target = 1450.0
        
        # MPC Karar Mekanizması
        try:
            fuel_cmd, fan_cmd = mpc.optimize(plant, current_target)
        except Exception as e:
            logger.error(f"MPC Optimizasyon Hatası (Adım {i}): {e}")
            break
            
        # Simülasyonda Adım At
        result = plant.step(fuel_cmd, fan_cmd)
        
        # Terminal Loglama (Her 120 dakikada bir - 2 saatte bir)
        # 5 sn adım için 120 dk = 1440 adım
        if i % 1440 == 0:
            current_min = (i * step_sec) / 60
            logger.info(
                f"Dakika: {int(current_min):4d} | "
                f"Sıcaklık: {result['Temperature']:.2f}°C | "
                f"Yakıt: {result['Fuel']:.2f} | "
                f"O2: {result['O2']:.2f}%"
            )

    # Verileri Kaydet
    if not os.path.exists("data"):
        os.makedirs("data")
        
    df = pd.DataFrame(plant.data)
    output_path = "data/pure_mpc_results_1450_fixed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Benchmark tamamlandı: {output_path}")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    try:
        # 1. Config Yükle
        cfg = load_config()
        logger.info("Config yüklendi.")

        # 2. Klasör yapısını kontrol et
        for folder in ["data", "models", "logs"]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # 3. MPC Benchmark Çalıştır (Sabit 1450)
        generate_pure_mpc_benchmark(cfg)

        # 4. RL Eğitimi (Hardware hatasını config'e eklediysen burası çalışır)
        # Eğer sadece MPC görmek istiyorsan alt satırı yorum satırı yapabilirsin.
        # train_rl(cfg)

    except FileNotFoundError:
        logger.error("config.yaml bulunamadı!")
    except Exception as e:
        logger.error(f"Ana döngüde kritik hata: {e}")
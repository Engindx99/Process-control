import os
import logging
import pandas as pd
import yaml
import numpy as np
import time

from src.dt.dt import RotaryKilnPlant
from src.mpc.mpc import MPC 
from src.filters.kalman import SelectiveKalmanFilter

# LOGGING AYARLARI
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_mpc_benchmark(config):
    # Çıktı klasörü kontrolü
    output_path = config.get("paths", {}).get("output_csv", "data/mpc_results.csv")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    logger.info("--- [MPC] Benchmark Simülasyonu Başlatılıyor... ---")
    
    # 1. Bileşenleri Başlatma
    plant = RotaryKilnPlant(seed=config["plant"].get("seed", 42))
    obs = plant.reset() 
    
    # MPC ve Kalman'ı başlat
    mpc = MPC("config.yaml") 
    kf = SelectiveKalmanFilter(config) 
    
    # 2. Zaman Planlaması
    step_sec = config["system"].get("step_duration_sec", 5)
    # 24 saatlik simülasyon (17280 adım)
    total_steps = int(24 * 3600 / step_sec) 

    logger.info(f"Parametreler: {total_steps} Adım | Adım Süresi: {step_sec}s | Hedef: {config['system']['setpoint']}°C")

    # Performans ölçümü
    start_sim_time = time.time()
    last_report_time = start_sim_time

    for i in range(total_steps):
        # --- FILTRELEME ADIMI ---
        # Sensör gürültüsünü temizliyoruz
        f_o2, f_press = kf.update(obs['o2'], obs['pressure'])
        
        # --- KONTROL ADIMI (MPC) ---
        try:
            # MPC'ye plant nesnesini gönderiyoruz. 
            # MPC içindeki 'deepcopy' işlemi bitince plant.data listesini 
            # temizlediğimizden emin olmalısın ki StepTime şişmesin.
            fuel_cmd, fan_cmd = mpc.get_action(plant) 
            
        except Exception as e:
            logger.error(f"MPC Kritik Hatası (Adım {i}): {e}", exc_info=True)
            break
        
        # --- PLANT ADIMI (GERÇEK DÜNYA) ---
        obs = plant.step(fuel_cmd, fan_cmd)
        
        # Periyodik Raporlama (Her 120 adım = 10 Dakika simülasyon zamanı)
        if i % 120 == 0 and i > 0:
            current_real_time = time.time()
            avg_step_perf = (current_real_time - last_report_time) / 120
            last_report_time = current_real_time
            
            logger.info(
                f"Adım {i:5d} | T: {obs['temp']:6.1f}°C | "
                f"Yakıt: {fuel_cmd:5.2f} | Fan: {fan_cmd:6.1f} | "
                f"O2: {obs['o2']:4.2f} | StepTime: {avg_step_perf:.3f}s"
            )

    # 3. Veriyi Kaydetme
    # DataFrame oluştururken sütun isimlerini temizle
    df = pd.DataFrame(plant.data)
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(output_path, index=False)
    
    total_duration_min = (time.time() - start_sim_time) / 60
    logger.info(f"--- [BAŞARI] Simülasyon {total_duration_min:.2f} dakikada bitti. ---")
    logger.info(f"Sonuçlar kaydedildi: {output_path}")

if __name__ == "__main__":
    try:
        # main.py içindeki ilgili satırı bul ve encoding='utf-8' ekle:
        with open("config.yaml", "r", encoding="utf-8") as f:
           cfg = yaml.safe_load(f)
        
        # Simülasyonu başlat
        generate_mpc_benchmark(cfg)
        
    except FileNotFoundError:
        logger.error("Hata: 'config.yaml' dosyası bulunamadı!")
    except Exception as e:
        logger.error(f"Sistem Başlatılamadı: {e}", exc_info=True)
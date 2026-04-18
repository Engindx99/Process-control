import os
import sys
import yaml
import pandas as pd
import numpy as np
import multiprocessing
import logging

# Kendi modüllerimiz
from src.rl.rl import train_rl
from src.mpc.mpc import MPC
from src.dt.dt import RotaryKilnDigitalTwin

# =========================================================
# LOGGER YAPILANDIRMASI
# =========================================================
logger = logging.getLogger("ORCHESTRATOR")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =========================================================
# ALTYAPI VE DOSYA KONTROLÜ
# =========================================================
def check_and_prepare_infrastructure():
    dirs = ["models", "data", "logs", "experiments/plots", "models/checkpoints"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    critical_files = [
        "src/dt/dt.py",
        "src/rl/rl.py",
        "src/mpc/mpc.py",
        "config.yaml"
    ]

    missing = [f for f in critical_files if not os.path.exists(f)]
    if missing:
        logger.error(f"Kritik dosyalar eksik: {missing}")
        sys.exit(1)

    logger.info("Altyapı hazır, dosyalar doğrulandı.")

# =========================================================
# CONFIG YÜKLEME
# =========================================================
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        logger.info("Config dosyası başarıyla yüklendi.")
        return cfg

# =========================================================
# MPC BENCHMARK (Opsiyonel - Gürültü Analizi İçin)
# =========================================================
def generate_pure_mpc_benchmark(config, path="data/pure_mpc_results.csv"):
    if os.path.exists(path):
        logger.info(f"MPC benchmark zaten mevcut, geçiliyor... -> {path}")
        return

    logger.info("MPC Benchmark başlatılıyor (Gürültü analizi)...")
    plant = RotaryKilnDigitalTwin(seed=42)
    mpc = MPC(config)
    history = []

    for i in range(3000):
        u_fuel, u_fan = mpc.optimize(plant)
        result = plant.step(u_fuel, u_fan)
        
        temp = result["Temperature"]
        o2 = result["O2"]
        error = temp - config["system"]["setpoint"]

        history.append({
            "step": i, "temp": temp, "fuel": u_fuel, 
            "fan": u_fan, "o2": o2, "error": error
        })

        if i % 500 == 0:
            logger.info(f"[MPC Benchmark] Step: {i}, Temp: {temp:.2f}, Error: {error:.2f}")

    pd.DataFrame(history).to_csv(path, index=False)
    logger.info("MPC Benchmark tamamlandı ve kaydedildi.")

# =========================================================
# ANA ÇALIŞTIRICI (MAIN)
# =========================================================
if __name__ == "__main__":
    # Windows sistemlerde multiprocessing hatasını önlemek için kritik
    multiprocessing.freeze_support()
    
    # 1. Hazırlık
    check_and_prepare_infrastructure()
    config = load_config()
    
    # 2. Benchmark (Eğer data/pure_mpc_results.csv yoksa çalışır)
    generate_pure_mpc_benchmark(config)

    # 3. RL EĞİTİMİNİ BAŞLAT
    # Bu fonksiyon src/rl/rl.py içindedir ve SubprocVecEnv, make_env, PPO 
    # gibi tüm süreçleri config'e göre otomatik yönetir.
    try:
        logger.info("Hibrit (MPC+RL) Eğitim süreci başlatılıyor...")
        train_rl(config)
        logger.info("Sistem eğitimi başarıyla tamamladı.")
        
    except KeyboardInterrupt:
        logger.warning("Eğitim kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.error(f"Sistem hatası: {e}")
        raise
import os
import sys
import yaml
import pandas as pd
import numpy as np
import multiprocessing
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# =========================================================
# LOGGER
# =========================================================
logger = logging.getLogger("ORCHESTRATOR")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =========================================================
# INFRA CHECK
# =========================================================
def check_and_prepare_infrastructure():
    dirs = ["models", "data", "experiments/plots", "models/checkpoints"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    critical_files = [
        "src/dt/dt.py",
        "src/rl/rl.py",
        "src/mpc/mpc.py"
    ]

    missing = [f for f in critical_files if not os.path.exists(f)]
    if missing:
        logger.error(f"Missing critical files: {missing}")
        sys.exit(1)

    logger.info("Infrastructure ready")

# =========================================================
# PATH & IMPORTS
# =========================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rl.rl import make_env, ResidualKilnEnv
from src.mpc.mpc import MPC
from src.dt.dt import RotaryKilnDigitalTwin

# =========================================================
# CONFIG
# =========================================================
def load_config():
    paths = ["config.yaml", "config/config.yaml"]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                logger.info(f"Config loaded: {p}")
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found")

# =========================================================
# MPC BENCHMARK (FIXED)
# =========================================================
def generate_pure_mpc_benchmark(config, path="data/pure_mpc_results.csv"):
    if os.path.exists(path):
        logger.info(f"MPC benchmark exists -> {path}")
        return

    logger.info("Starting MPC benchmark")
    plant = RotaryKilnDigitalTwin(seed=42)
    mpc = MPC(config)
    history = []

    for i in range(3000):
        u_fuel, u_fan = mpc.optimize(plant)
        
        # --- DICTIONARY FIX ---
        result = plant.step(u_fuel, u_fan)
        temp = result["Temperature"]
        o2 = result["O2"]
        # ----------------------

        error = temp - config["system"]["setpoint"]

        history.append({
            "step": i,
            "temp": temp,
            "fuel": u_fuel,
            "fan": u_fan,
            "o2": o2,
            "error": error
        })

        if i % 100 == 0:
            logger.info(
                f"[MPC] step={i} temp={temp:.2f} error={error:.2f} o2={o2:.2f}"
            )

    pd.DataFrame(history).to_csv(path, index=False)
    logger.info("MPC benchmark completed")

# =========================================================
# TEST LOOP
# =========================================================
def run_final_test(model, config, n_steps=3000):
    logger.info(f"Running hybrid test: {n_steps} steps")
    test_env = ResidualKilnEnv(config)
    obs, _ = test_env.reset(seed=42)
    history = []

    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)

        # Verileri doğrudan env.plant üzerinden alıyoruz (En günceli oradadır)
        history.append({
            "step": i,
            "temp": test_env.plant.temp,
            "fuel": test_env.plant.fuel,
            "fan": test_env.plant.fan,
            "o2": test_env.plant.o2,
            "reward": reward
        })

        if i % 500 == 0:
            logger.info(f"[TEST] step={i} temp={test_env.plant.temp:.2f}")

        if terminated or truncated:
            logger.warning(f"Episode ended at step {i}")
            break

    return pd.DataFrame(history)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # Windows için çoklu işlem desteği
    multiprocessing.freeze_support()
    
    check_and_prepare_infrastructure()
    config = load_config()

    model_path = config["paths"]["model_save_path"]
    total_steps = config["rl"].get("total_timesteps")
    seed = config["rl"].get("seed", 42)
    np.random.seed(seed)

    # 1. MPC Benchmark çalıştır
    generate_pure_mpc_benchmark(config)

    # 2. Ortam kurulumu (Paralel CPU)
    num_cpu = config["hardware"]["num_cpu"]
    logger.info(f"Starting distributed training: {num_cpu} CPUs")
    env = SubprocVecEnv([make_env(config, i) for i in range(num_cpu)])

    # 3. Model Yükleme veya Oluşturma
    if os.path.exists(model_path + ".zip"):
        logger.info(f"Loading existing model: {model_path}")
        model = PPO.load(model_path, env=env)
        model.learning_rate = float(config["rl"]["learning_rate"])
    else:
        logger.info("Creating new PPO model for 50k test")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=float(config["rl"]["learning_rate"]),
            n_steps=config["rl"]["n_steps"],
            batch_size=config["rl"]["batch_size"],
            gamma=config["rl"]["gamma"],
            device=config["hardware"]["device"]
        )

    # 4. Eğitim
    try:
        logger.info(f"Training session started for {total_steps} steps...")
        model.learn(total_timesteps=total_steps)
        model.save(model_path)
        logger.info(f"Model successfully saved -> {model_path}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user, saving progress...")
        model.save(model_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # 5. Final Testi ve Kayıt
    try:
        df_test = run_final_test(model, config)
        df_test.to_csv("data/training_results.csv", index=False)
        logger.info("Final hybrid test results saved to data/training_results.csv")
    finally:
        env.close()
        logger.info("System shutdown complete.")
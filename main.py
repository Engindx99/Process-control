import os
import yaml
import pandas as pd
from stable_baselines3 import PPO

from src.rl.rl import train_model, run_test

# --- CONFIG ---
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    model_path = cfg['paths']['model_name']
    zip_path = model_path if model_path.endswith(".zip") else model_path + ".zip"

    # --- LOAD / TRAIN ---
    if os.path.exists(zip_path):
        print("✅ Model yükleniyor...")
        model = PPO.load(model_path, device=cfg['hardware']['device'])
    else:
        print("🚀 Eğitim başlıyor...")
        model = train_model(cfg)
        model.save(model_path)
        print("💾 Model kaydedildi")

    # --- TEST ---
    print("📊 Test başlıyor...")
    history, env = run_test(model, cfg)

    df = pd.DataFrame(history)
    df.to_csv(cfg['paths']['results_csv'], index=False)

    env.mpc.save(cfg['paths']['log_json'])

    print(" Bitti")
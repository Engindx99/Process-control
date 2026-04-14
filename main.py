import os
import sys
import yaml
import pandas as pd
import numpy as np
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# 1. MODÜL VE ALTYAPI KONTROLÜ
def check_and_prepare_infrastructure():
    """Gerekli klasörleri oluşturur ve kritik dosyaları kontrol eder."""
    dirs = ["models", "data", "experiments/plots"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    critical_files = ["src/digital_twin/dt.py", "src/rl/rl.py", "src/mpc/mpc.py"]
    missing = [f for f in critical_files if not os.path.exists(f)]
    if missing:
        print(f"❌ KRİTİK HATA: Kaynak dosyalar eksik: {missing}")
        sys.exit(1)
    print("✅ Altyapı ve klasörler hazır.")

try:
    from src.rl.rl import make_env, ResidualKilnEnv
    from src.mpc.mpc import MPC
    from src.digital_twin.dt import RotaryKilnDigitalTwin
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.rl.rl import make_env, ResidualKilnEnv
    from src.mpc.mpc import MPC
    from src.digital_twin.dt import RotaryKilnDigitalTwin

def load_config():
    paths = ["config.yaml", "config/config.yaml"]
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r') as f: return yaml.safe_load(f)
    raise FileNotFoundError("❌ config.yaml bulunamadı!")

def generate_pure_mpc_benchmark(config, path="data/pure_mpc_results.csv"):
    """Eğer yoksa, kıyaslama için saf MPC verilerini üretir."""
    if os.path.exists(path):
        print(f"✅ Saf MPC verisi mevcut: {path}")
        return
    
    print("🚀 Saf MPC Benchmark süreci başlatıldı (Veri üretiliyor)...")
    plant = RotaryKilnDigitalTwin()
    mpc = MPC(config)
    history = []
    
    for i in range(3000):
        u_mpc = mpc.optimize(plant)
        temp, o2, _ = plant.step(u_mpc, 1000.0)
        error = temp - 1450
        
        history.append({
            "step": i, 
            "temp": temp, 
            "fuel": u_mpc, 
            "o2": o2, 
            "error": error
        })

        # --- LOGLAMA EKRANI ---
        if i % 100 == 0:  # Her 100 adımda bir durum raporu ver
            status = "YÜKSELİYOR" if error < -5 else "DENGELİ" if abs(error) < 5 else "YÜKSEK"
            print(f"Step: {i:4d} | Sıcaklık: {temp:7.2f}°C | Hata: {error:6.2f}°C | Yakıt: {u_mpc:5.2f} | Durum: {status}")
    
    pd.DataFrame(history).to_csv(path, index=False)
    print(f"\n✅ Saf MPC benchmark tamamlandı ve '{path}' dosyasına kaydedildi.")

def run_final_test(model, config, n_steps=3000):
    print(f"📊 Hibrit Test sürüşü başlatılıyor ({n_steps} adım)...")
    test_env = ResidualKilnEnv(config)
    obs, _ = test_env.reset()
    history = []
    
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        history.append({
            "step": i,
            "temp": test_env.plant.temp,
            "fuel": test_env.plant.fuel,
            "o2": test_env.plant.o2,
            "reward": reward,
            "residual_fuel": float(action[0]) * config['rl']['action_limit']
        })
        if terminated or truncated: break
    return pd.DataFrame(history)

# =================================================================
# ANA AKIŞ
# =================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    check_and_prepare_infrastructure()
    
    config = load_config()
    model_path = config['paths']['model_save_path']
    total_steps = config['rl'].get('total_timesteps', 100000)

    # 1. Eksikse Saf MPC Verisini Üret (Benchmark)
    generate_pure_mpc_benchmark(config)

    # 2. Paralel Ortamı Kur
    env = SubprocVecEnv([make_env(config) for _ in range(config['hardware']['num_cpu'])])

    # 3. Model Kontrol / Eğitim
    if os.path.exists(model_path + ".zip"):
        print(f"✅ Model bulundu, yükleniyor: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("🔍 Model bulunamadı, sıfırdan eğitime başlanıyor...")
        model = PPO("MlpPolicy", env, verbose=1, 
                    learning_rate=config['rl']['learning_rate'],
                    n_steps=config['rl']['n_steps'],
                    batch_size=config['rl']['batch_size'],
                    gamma=config['rl']['gamma'],
                    device=config['hardware']['device'])
        try:
            model.learn(total_timesteps=total_steps)
            model.save(model_path)
        except KeyboardInterrupt:
            print("\n🛑 Eğitim durduruldu, mevcut durum kaydediliyor...")
            model.save(model_path)

    # 4. Hibrit Sonuçları Üret
    try:
        df_hybrid = run_final_test(model, config)
        df_hybrid.to_csv("data/training_results.csv", index=False)
        print("📈 Hibrit test verileri kaydedildi: data/training_results.csv")
        print("✨ Tüm veriler hazır. Artık karşılaştırma grafiğini çizebilirsin!")
    finally:
        env.close()
        print("🏁 Sistem kapatıldı.")
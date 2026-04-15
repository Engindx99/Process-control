import os
import sys
import yaml
import pandas as pd
import numpy as np
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# 1. ALTYAPI VE KLASÖR HAZIRLIĞI
def check_and_prepare_infrastructure():
    dirs = ["models", "data", "experiments/plots"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    critical_files = ["src/dt/dt.py", "src/rl/rl.py", "src/mpc/mpc.py"]
    missing = [f for f in critical_files if not os.path.exists(f)]
    if missing:
        print(f"❌ KRİTİK HATA: Kaynak dosyalar eksik: {missing}")
        sys.exit(1)
    print("✅ Altyapı hazır. Eğitim ortamı kuruluyor...")

# Modül yollarını ekle ve import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.rl.rl import make_env, ResidualKilnEnv
    from src.mpc.mpc import MPC
    from src.dt.dt import RotaryKilnDigitalTwin
except ImportError as e:
    print(f"❌ Import Hatası: {e}")
    sys.exit(1)

def load_config():
    paths = ["config.yaml", "config/config.yaml"]
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r') as f: return yaml.safe_load(f)
    raise FileNotFoundError("❌ config.yaml bulunamadı!")

# 2. BENCHMARK (SAF MPC) ÜRETİMİ
def generate_pure_mpc_benchmark(config, path="data/pure_mpc_results.csv"):
    """
    Saf MPC performansını ölçer. 
    Not: Bu kısım ana süreçte (tek çekirdek) çalıştığı için loglar açık kalabilir.
    """
    if os.path.exists(path):
        print(f"✅ Saf MPC verisi mevcut, benchmark atlanıyor: {path}")
        return
    
    print("\n🚀 MPC Benchmark Başlatıldı (Yakıt & Fan Optimizasyonu)...")
    plant = RotaryKilnDigitalTwin()
    mpc = MPC(config)
    history = []
    
    for i in range(3000):
        # MPC'den aksiyon al
        u_fuel, u_fan = mpc.optimize(plant)
        
        # Dijital ikiz üzerinde uygula
        temp, o2, _ = plant.step(u_fuel, u_fan)
        error = temp - 1450.0
        
        history.append({
            "step": i, "temp": temp, "fuel": u_fuel, 
            "fan": u_fan, "o2": o2, "error": error
        })

        if i % 100 == 0:
            print(f"Benchmark Step: {i:4d} | T: {temp:7.2f}°C | Hata: {error:6.2f} | O2: %{o2:4.2f}")
    
    pd.DataFrame(history).to_csv(path, index=False)
    print("✅ Benchmark tamamlandı.\n")

# 3. TEST SÜRÜŞÜ (HİBRİT MODEL)
def run_final_test(model, config, n_steps=3000):
    print(f"📊 Hibrit (MPC + RL) Test sürüşü başlatılıyor ({n_steps} adım)...")
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
            "fan": test_env.plant.fan,
            "o2": test_env.plant.o2,
            "reward": reward
        })
        
        if i % 500 == 0:
            print(f"Test Step: {i:4d} | Hibrit T: {test_env.plant.temp:7.2f}°C")
            
        if terminated or truncated: break
        
    return pd.DataFrame(history)

# 4. ANA ÇALIŞTIRICI
if __name__ == "__main__":
    multiprocessing.freeze_support()
    check_and_prepare_infrastructure()
    
    config = load_config()
    model_path = config['paths']['model_save_path']
    total_steps = config['rl'].get('total_timesteps', 200000)

    # Önce Saf MPC Benchmark üret
    generate_pure_mpc_benchmark(config)

    # Paralel ortamı hazırla (Hız için kritik!)
    num_cpu = config['hardware']['num_cpu']
    print(f"⚙️  {num_cpu} çekirdek üzerinde paralel eğitim başlıyor...")
    env = SubprocVecEnv([make_env(config, i) for i in range(num_cpu)])

    # Model Kontrol ve Eğitim
    if os.path.exists(model_path + ".zip"):
        print(f"♻️  Mevcut model yükleniyor: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("🆕 Yeni model oluşturuluyor...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, # SB3 Tablo logları için açık kalsın
            learning_rate=float(config['rl']['learning_rate']),
            n_steps=config['rl']['n_steps'],
            batch_size=config['rl']['batch_size'],
            gamma=config['rl']['gamma'],
            device=config['hardware']['device']
        )
    
    try:
        if total_steps > 0:
            print(f"🏋️  Eğitim süreci aktif: {total_steps} adım.")
            model.learn(total_timesteps=total_steps)
            model.save(model_path)
            print(f"💾 Model kaydedildi: {model_path}")
    except KeyboardInterrupt:
        print("\n🛑 Kullanıcı tarafından durduruldu. Kaydediliyor...")
        model.save(model_path)

    # Sonuçları Kaydet
    try:
        df_hybrid = run_final_test(model, config)
        df_hybrid.to_csv("data/training_results.csv", index=False)
        print("📈 Hibrit sonuçlar 'data/training_results.csv' dosyasına yazıldı.")
    finally:
        env.close()
        print("🏁 İşlem başarıyla sonlandırıldı.")
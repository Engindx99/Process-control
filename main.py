import os
import sys
import yaml
import pandas as pd
import numpy as np
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# 1. MODÜL İMPORTLARI VE YOL KONTROLÜ
try:
    from src.rl.rl import make_env, ResidualKilnEnv
except ModuleNotFoundError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    try:
        from src.rl.rl import make_env, ResidualKilnEnv
    except ModuleNotFoundError:
        print("❌ HATA: 'src' klasörü bulunamadı. Lütfen 'Rotary_klin' dizininde olduğunuzdan emin olun.")
        sys.exit(1)

def load_config():
    """Yapılandırma dosyasını kökte veya config/ klasöründe arar."""
    possible_paths = ["config.yaml", "config/config.yaml"]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"❌ Yapılandırma dosyası bulunamadı!")

def run_final_test(model, config, n_steps=3000):
    """Eğitim bittiğinde ajanın performansını test eder ve verileri toplar."""
    print(f"📊 Test sürüşü başlatılıyor ({n_steps} adım)...")
    test_env = ResidualKilnEnv(config)
    obs, _ = test_env.reset()
    
    history = []
    for i in range(n_steps):
        # Deterministic=True: Eğitimdeki rastgeleliği kapat, en iyi hamleyi yap
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
# ANA AKIŞ (MAIN EXECUTION)
# =================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1. Yapılandırmayı Yükle
    try:
        config = load_config()
    except Exception as e:
        print(f"❌ Yapılandırma hatası: {e}")
        sys.exit(1)

    total_steps = config['rl'].get('total_timesteps', 100000)
    model_path = config['paths']['model_save_path']
    
    print(f"🚀 --- Döner Fırın Hibrit Kontrol Sistemi ---")
    print(f"📍 Hedef: {total_steps} Adım | Çekirdek: {config['hardware']['num_cpu']}")

    # 2. Klasörleri Otomatik Oluştur
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments/plots", exist_ok=True)

    # 3. Paralel Ortamları Başlat
    env = SubprocVecEnv([make_env(config) for _ in range(config['hardware']['num_cpu'])])

    # 4. Model Kontrolü ve Koşullu Eğitim
    if os.path.exists(model_path + ".zip"):
        # MODEL VARSA: Yükle ve eğitimi atla
        print(f"✅ Kayıtlı model bulundu: {model_path}.zip")
        print("🚀 Eğitim atlanıyor, doğrudan teste geçiliyor...")
        model = PPO.load(model_path, env=env)
    else:
        # MODEL YOKSA: PPO'yu tanımla ve eğit
        print(f"🔍 Model bulunamadı. Sıfırdan eğitime başlanıyor...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=config['rl']['learning_rate'],
            n_steps=config['rl']['n_steps'],
            batch_size=config['rl']['batch_size'],
            gamma=config['rl']['gamma'],
            device=config['hardware']['device']
        )
        try:
            print(f"🧠 Sinir ağı eğitiliyor... Lütfen bekleyin.")
            model.learn(total_timesteps=total_steps)
            model.save(model_path)
            print(f"✅ Eğitim tamamlandı ve model kaydedildi.")
        except KeyboardInterrupt:
            print("\n🛑 Eğitim durduruldu. Mevcut durum kaydediliyor...")
            model.save(model_path)

    # 5. SONUÇLARIN KAYDI VE TEST (3000 ADIM)
    try:
        # Adım sayısını config'den al (total_steps: 3000)
        test_duration = config['simulation'].get('total_steps', 3000)
        df_results = run_final_test(model, config, n_steps=test_duration)
        
        # CSV Kaydı
        results_path = "experiments/plots/training_results.csv"
        df_results.to_csv(results_path, index=False)
        
        print(f"📈 Detaylı test verileri '{results_path}' dosyasına kaydedildi.")
        print(f"✨ İşlem başarıyla tamamlandı.")

    except Exception as e:
        print(f"❌ Test sırasında bir hata oluştu: {e}")
    
    finally:
        env.close()
        print("🏁 Sistem kapatıldı.")
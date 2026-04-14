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
    # Eğer PYTHONPATH ayarlanmadıysa manuel eklemeyi dene
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
    raise FileNotFoundError(f"❌ Yapılandırma dosyası bulunamadı! Aranan konumlar: {possible_paths}")

def run_final_test(model, config, n_steps=1000):
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
    # Windows Multiprocessing Desteği (Kritik!)
    multiprocessing.freeze_support()

    # 1. Yapılandırmayı Yükle
    try:
        config = load_config()
    except Exception as e:
        print(e)
        sys.exit(1)

    total_steps = config['rl'].get('total_timesteps', 100000)
    
    print(f"🚀 --- Döner Fırın Hibrit Eğitimi Hazırlığı ---")
    print(f"📍 Hedef: {total_steps} Adım | Çekirdek: {config['hardware']['num_cpu']}")
    print(f"📍 Cihaz: {config['hardware']['device']}")

    # 2. Klasörleri Otomatik Oluştur
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 3. Paralel Ortamları (SubprocVecEnv) Başlat
    # Her çekirdek için bir ResidualKilnEnv örneği oluşturulur
    env = SubprocVecEnv([make_env(config) for _ in range(config['hardware']['num_cpu'])])

    # 4. PPO Algoritmasını Tanımla
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

    # 5. Daha Önce Kaydedilmiş Model Varsa Yükle (Eğitimi Sürdür)
    model_path = config['paths']['model_save_path']
    if os.path.exists(model_path + ".zip"):
        print(f"🔄 Mevcut model ({model_path}.zip) bulundu, üzerine eğitim devam ediyor...")
        model = PPO.load(model_path, env=env)
    
    # 6. EĞİTİM DÖNGÜSÜ
    try:
        print(f"🧠 Sinir ağı eğitiliyor... Lütfen bekleyin.")
        model.learn(total_timesteps=total_steps)
        
        # Modeli Kaydet
        model.save(model_path)
        print(f"✅ Model başarıyla kaydedildi: {model_path}")

        # 7. SONUÇLARIN KAYDI VE TEST
        df_results = run_final_test(model, config)
        df_results.to_csv("data/training_results.csv", index=False)
        print("📈 Detaylı test verileri 'data/training_results.csv' dosyasına kaydedildi.")

    except KeyboardInterrupt:
        print("\n🛑 Eğitim kullanıcı tarafından durduruldu (Ctrl+C). Mevcut ilerleme kaydediliyor...")
        model.save(model_path)
    
    except Exception as e:
        print(f"❌ Beklenmedik bir hata oluştu: {e}")
    
    finally:
        # Alt işlemleri kapat ve belleği temizle
        env.close()
        print("🏁 İşlem tamamlandı.")
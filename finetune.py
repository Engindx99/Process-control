import yaml
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from src.rl.rl import ResidualKilnEnv

def load_config(path="config.yaml"):
    # UnicodeDecodeError (0x9e) hatasını engellemek için utf-8 ekledik
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_finetune():
    cfg = load_config()
    
    # --- FINE-TUNE & HIZ AYARLARI ---
    LR_FINETUNE = 5e-5  
    TOTAL_STEPS = 300_000 
    
    # KeyError: 'runtime' hatasını engellemek için güvenli atama
    if "runtime" not in cfg:
        cfg["runtime"] = {}
    cfg["runtime"]["render_mode"] = None # Görselleştirmeyi kapat, FPS uçsun.

    # 1. Ortamı Kur (Hız için paralel çekirdek kullanımı)
    num_cpu = cfg["hardware"].get("num_cpu", 4)
    env = SubprocVecEnv([lambda i=i: ResidualKilnEnv(cfg) for i in range(num_cpu)])
    env = VecMonitor(env)

    # 2. Modeli Yükle (CUDA Devrede)
    model_path = cfg["paths"]["model_save_path"]
    
    if os.path.exists(model_path):
        print(f"--- [Fine-Tune] {model_path} yükleniyor... ---")
        model = PPO.load(
            model_path, 
            env=env, 
            device=cfg.get("hardware", {}).get("device", "cuda"),
            custom_objects={"learning_rate": LR_FINETUNE}
        )
    else:
        print("HATA: Kaynak model bulunamadı!")
        return

    # 3. İnce Ayar Eğitimini Başlat
    print(f"--- FPS İyileştirildi. Eğitim başlıyor: {TOTAL_STEPS} adım ---")
    
    # progress_bar=True ile terminal trafiğini azaltıp hızı artırıyoruz
    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=False, progress_bar=True)
    
    # 4. Kaydet (Eskisinin üzerine yazmaz, rafine halini ayırır)
    refined_path = model_path.replace(".zip", "_refined.zip")
    model.save(refined_path)
    print(f"--- İşlem Tamam! Yeni model: {refined_path} ---")

if __name__ == "__main__":
    run_finetune()
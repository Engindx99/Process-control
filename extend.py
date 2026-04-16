import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.rl.rl import make_env 

def extend_training():
    # 1. Yapılandırmayı Yükle
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Çoklu Çekirdek Ortamını Oluştur (n_envs=11)
    num_cpu = 11 # İşlemci çekirdek sayına göre burayı 8, 11 veya 12 yapabilirsin
    print(f"🔥 {num_cpu} çekirdek üzerinde ortamlar hazırlanıyor...")
    
    env = SubprocVecEnv([make_env(cfg, seed=42, rank=i) for i in range(num_cpu)])

    # 3. Mevcut Modeli Yükle
    model_path = "models/ppo_kiln_v5_pro"
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env)
        print(f"✅ Model yüklendi. Çekirdek sayısı: {num_cpu}")
    else:
        print("❌ Hata: v5_pro modeli bulunamadı!")
        return

    # 4. Hızı Kesin Olarak Sabitle (Garanti Yöntem)
    new_lr = 5e-5
    model.learning_rate = new_lr
    # Optimizer'ı yeni hıza zorla (Çalışırken değişmesi için)
    model.policy.optimizer.param_groups[0]['lr'] = new_lr

    print(f"🎯 Hedeflenen Öğrenme Hızı: {new_lr}")
    print(f"🚀 {num_cpu} çekirdek ile toplam 300.000 adım eğitim başlıyor...")
    
    # 5. Eğitimi Başlat
    model.learn(
        total_timesteps=300000, 
        reset_num_timesteps=False,
        progress_bar=True
    )

    # 6. Kaydet
    new_model_name = "models/ppo_kiln_v5_pro_extended_3k"
    model.save(new_model_name)
    print(f"💾 Yeni model başarıyla kaydedildi: {new_model_name}")

if __name__ == "__main__":
    extend_training()
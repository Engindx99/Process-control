import os
import yaml
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Kendi modüllerinden importlar
from src.physics.digital_twin import IndustrialRotaryKiln
from src.control.mpc import AdvancedRotaryKilnMPC

# --- Config Yükle ---
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ----------------------------------
# 1. ANLIK TAKİP CALLBACK SINIFI
# ----------------------------------
class SimpleProgressCallback(BaseCallback):
    def __init__(self, check_freq=10):
        super().__init__()
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            reward = np.mean(self.locals['rewards']) if 'rewards' in self.locals else 0
            print(f">>> Toplam Adım: {self.num_timesteps} | Ortalama Ödül: {reward:.2f}")
        return True

# ----------------------------------
# 2. ENVIRONMENT (Config Destekli)
# ----------------------------------
class RotaryKilnEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.kiln = IndustrialRotaryKiln()
        self.mpc = AdvancedRotaryKilnMPC()

        self.action_space = gym.spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=2000, shape=(4,), dtype=np.float32)

        self.feed = cfg['env']['feed_base']
        self.target_base = cfg['env']['target_base']

    def step(self, action):
        offset = action[0]
        dynamic_target = self.target_base + offset
        
        temp_meas, o2_meas = self.kiln.measure()
        
        def dynamic_tvp(t_now):
            tvp = self.mpc.mpc.get_tvp_template()
            tvp['_tvp', :, 'temp_setpoint'] = dynamic_target
            tvp['_tvp', :, 'material_feed'] = self.feed
            return tvp
        
        self.mpc.mpc.set_tvp_fun(dynamic_tvp)
        fuel, fan = self.mpc.step(temp_meas)

        obs_dict = self.kiln.step(fuel_cmd=fuel, fan_cmd=fan, feed=self.feed)
        temp, o2 = obs_dict["temp"], obs_dict["oxygen"]

        # --- GÜNCELLEME: KARESEL ÖDÜL HESAPLAMASI ---
        o2_error = abs(o2 - 4.0)
        
        # Oksijen hatasının karesini alarak cezayı eksponansiyel hale getirdik
        reward = -(
            abs(temp - dynamic_target) * float(cfg['reward']['temp_weight']) +
            (o2_error**2) * float(cfg['reward']['o2_weight']) +
            float(cfg['reward']['fuel_weight']) * (fuel**2)
        )

        state = np.array([temp, o2, fuel, fan], dtype=np.float32)
        return state, float(reward), False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.kiln = IndustrialRotaryKiln()
        self.mpc = AdvancedRotaryKilnMPC()
        t_m, o2_m = self.kiln.measure()
        state = np.array([t_m, o2_m, 0.0, 50.0], dtype=np.float32)
        return state, {}

def make_env():
    return RotaryKilnEnv()

# ----------------------------------
# 3. ANA ÇALIŞTIRMA BLOĞU
# ----------------------------------
if __name__ == "__main__":
    output_dir = cfg['project']['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_cpu = cfg['rl']['num_cpu']
    print(f"\n🚀 {num_cpu} Çekirdek ve YAML Config ile Eğitim Başlatılıyor...")

    if num_cpu > 1:
        try:
            env = SubprocVecEnv([make_env for _ in range(num_cpu)], start_method='spawn')
        except Exception as e:
            print(f"Paralel hata: {e}. DummyVecEnv'e dönülüyor.")
            env = DummyVecEnv([make_env])
    else:
        env = DummyVecEnv([make_env])

    model_path = os.path.join(output_dir, "advanced_kiln_model")

    # DİKKAT: Yeni fizik ve yeni ödül yapısı için mevcut modeli SİLİP sıfırdan başlatman önerilir.
    if os.path.exists(model_path + ".zip"):
        print("\n--- Mevcut Model Yükleniyor ---")
        model = PPO.load(model_path, env=env)
    else:
        print("\n--- Yeni Eğitim Başlıyor ---")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            n_steps=cfg['rl']['n_steps'], 
            learning_rate=float(cfg['rl']['learning_rate']),
            tensorboard_log=cfg['project']['log_dir']
        )
    
    callback = SimpleProgressCallback(check_freq=10)
    model.learn(total_timesteps=cfg['rl']['total_timesteps'], callback=callback)
    model.save(model_path)

    # 4. TEST VE KAYIT
    print("\n📊 Test Simülasyonu Kaydediliyor...")
    test_env = RotaryKilnEnv()
    obs, _ = test_env.reset()
    results = []

    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, _ = test_env.step(action)
        results.append({
            "adim": i,
            "sicaklik": obs[0],
            "oksijen": obs[1],
            "yakit": obs[2],
            "fan": obs[3],
            "hedef": cfg['env']['target_base'] + action[0]
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "simulasyon_verileri.csv"), index=False)
    print(f"\n✅ İşlem Tamamlandı. Çıktılar '{output_dir}' klasöründe.")
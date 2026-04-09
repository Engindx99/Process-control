import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Import yollarını kendi klasör yapına göre kontrol et
from src.control.mpc import AdvancedRotaryKilnMPC
from src.physics.digital_twin import IndustrialRotaryKiln 

class RotaryKilnEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # ---------------- SYSTEM ----------------
        self.mpc = AdvancedRotaryKilnMPC()
        self.plant = IndustrialRotaryKiln()

        # ---------------- RL SPACE ----------------
        # RL, ana hedefe (1400) küçük düzeltmeler yapar
        self.action_space = gym.spaces.Box(
            low=-20, high=20, shape=(1,), dtype=np.float32
        )

        # Gözlem Alanı (Observation): [Sıcaklık, Oksijen, Besleme Hızı, Mevcut Setpoint]
        # ÖNEMLİ: Shape 4 olmalı ki ajan hedefini bilsin
        self.observation_space = gym.spaces.Box(
            low=np.array([700, 0, 3.0, 1380]),
            high=np.array([1650, 10, 7.0, 1420]),
            dtype=np.float32
        )

        # ---------------- INITIALS ----------------
        self.base_setpoint = 1400.0
        self.current_setpoint = 1400.0
        self.feed_rate = 5.0

    def step(self, action):
        # 1) RL'nin belirlediği yeni dinamik hedef
        self.current_setpoint = self.base_setpoint + float(action[0])

        # 2) MPC'ye hedefi bildir (Bu kısım MPC'yi RL ile senkronize eder)
        def tvp_fun(t_now):
            tvp = self.mpc.mpc.get_tvp_template()
            tvp['_tvp', :, 'temp_setpoint'] = self.current_setpoint
            tvp['_tvp', :, 'material_feed'] = self.feed_rate
            return tvp
        self.mpc.mpc.set_tvp_fun(tvp_fun)

        # 3) Kontrol ve Fiziksel Simülasyon
        temp_meas, _ = self.plant.measure()
        fuel, fan = self.mpc.step(temp_meas)
        
        # Digital Twin step fonksiyonunu çağır
        obs_dict = self.plant.step(fuel_cmd=fuel, fan_cmd=fan, feed=self.feed_rate)
        
        new_temp = obs_dict["temp"]
        new_o2 = obs_dict["oxygen"]

        # 4) REWARD (Geliştirilmiş Ceza Sistemi)
        # Hata karesel (l2) olursa ajan hedefe daha sert yönelir
        error_penalty = (new_temp - self.current_setpoint)**2
        o2_penalty = abs(new_o2 - 4.0) * 35
        fuel_penalty = 0.05 * (fuel**2)

        reward = -(error_penalty * 0.1 + o2_penalty + fuel_penalty)

        # 5) Çevresel Değişim (Besleme hızı rastgele değişir)
        self.feed_rate = np.clip(
            self.feed_rate + np.random.normal(0, 0.05), 3.0, 7.0
        )

        # Gözlem: [sıcaklık, oksijen, besleme, hedef]
        obs = np.array([new_temp, new_o2, self.feed_rate, self.current_setpoint], dtype=np.float32)

        return obs, float(reward), False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.plant = IndustrialRotaryKiln()
        self.current_setpoint = self.base_setpoint
        self.feed_rate = 5.0

        temp_meas, o2_meas = self.plant.measure()
        obs = np.array([temp_meas, o2_meas, self.feed_rate, self.current_setpoint], dtype=np.float32)

        return obs, {}

# ---------------- TRAIN ----------------
if __name__ == "__main__":
    env = RotaryKilnEnv()

    # i7-12700H için optimize edilmiş hiper-parametreler
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-4, 
        n_steps=2048,
        batch_size=128,
        gamma=0.95, # Fırın yavaş bir sistemdir, kısa vadeye odaklanmak iyidir
        tensorboard_log="./ppo_kiln_logs/"
    )

    print("🚀 Hibrit Eğitim Başlıyor: RL + MPC + Digital Twin")
    model.learn(total_timesteps=60000)
    model.save("outputs/advanced_kiln_model")
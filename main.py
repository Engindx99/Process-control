import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import do_mpc
import gymnasium as gym
from stable_baselines3 import PPO

# ==========================================
# 1. MPC MODELİ VE KONTROLCÜSÜ (BEYİN)
# ==========================================
def setup_mpc():
    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)

    # Değişkenler
    temp = model.set_variable(var_type='_x',  var_name='temp', shape=(1,1))
    fuel = model.set_variable(var_type='_u',  var_name='fuel', shape=(1,1))
    t_set = model.set_variable(var_type='_tvp', var_name='temp_setpoint')
    feed  = model.set_variable(var_type='_tvp', var_name='material_feed')

    # Fiziksel Denklem: Isı Dengesi
    dT_dt = (fuel * 30000 - 0.05 * (temp - 25) - (feed * 1.1 * (temp - 100))) / 5500
    model.set_rhs('temp', dT_dt)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    
    # En güvenli ayar yöntemi (Versiyon uyumlu)
    mpc.settings.n_horizon = 15
    mpc.settings.t_step = 1.0
    mpc.settings.n_robust = 0
    mpc.settings.store_full_solution = True
    mpc.settings.supress_ipopt_output = True # Terminal kirliliğini engeller
    
    # Hedef Fonksiyonu: Hata Kareleri + Yakıt Maliyeti
    lterm = (model.x['temp'] - model.tvp['temp_setpoint'])**2
    mpc.set_objective(lterm=lterm, mterm=lterm)
    mpc.set_rterm(fuel=1e-2)

    # Operasyonel Sınırlar
    mpc.bounds['lower', '_u', 'fuel'] = 0.5
    mpc.bounds['upper', '_u', 'fuel'] = 12.0
    mpc.bounds['upper', '_x', 'temp'] = 1550.0

    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_now):
        for i in range(16):
            tvp_template['_tvp', i, 'temp_setpoint'] = 1400.0
            tvp_template['_tvp', i, 'material_feed'] = 5.0
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()
    return mpc

# ==========================================
# 2. RL ORTAMI (GYM WRAPPER)
# ==========================================
class RotaryKilnEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.mpc_controller = setup_mpc()
        self.action_space = gym.spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=2000, shape=(3,), dtype=np.float32)
        self.state = np.array([1200.0, 5.0, 5.0], dtype=np.float32)

    def step(self, action):
        current_temp = self.state[0]
        feed_rate = self.state[2]
        target_temp = 1400.0 + action[0]

        # MPC Karar Mekanizması
        self.mpc_controller.x0 = np.array([current_temp])
        fuel_suggestion = self.mpc_controller.make_step(np.array([current_temp]))[0][0]

        # Fiziksel Motor (Sayısal Simülasyon)
        new_temp = current_temp + (fuel_suggestion * 4.2) - (feed_rate * 1.6) + np.random.normal(0, 0.4)
        
        # Ödül Fonksiyonu (Verimlilik vs Performans)
        temp_penalty = abs(new_temp - target_temp) * 0.1
        fuel_penalty = fuel_suggestion * 0.2
        reward = -(temp_penalty + fuel_penalty)
        
        self.state = np.array([new_temp, fuel_suggestion, feed_rate], dtype=np.float32)
        return self.state, float(reward), False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([1200.0, 5.0, 5.0], dtype=np.float32)
        return self.state, {}

# ==========================================
# 3. ÇALIŞTIRMA VE RAPORLAMA
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("ROTARY KILN AI: ENDÜSTRİYEL KONTROL SİSTEMİ")
    print("="*50)
    
    env = RotaryKilnEnv()
    
    print("\nAjan Eğitiliyor (SB3 - PPO)...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=3000)
    print("Eğitim Tamamlandı. Test Başlıyor...")

    # Simülasyon Testi
    obs, _ = env.reset()
    history = {"temp": [], "fuel": [], "target": []}

    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, _ = env.step(action)
        history["temp"].append(obs[0])
        history["fuel"].append(obs[1])
        history["target"].append(1400.0 + action[0])

    # --- GRAFİK TASARIMI ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Sıcaklık Grafiği
    ax1.plot(history["temp"], color='crimson', linewidth=2.5, label="Fırın Sıcaklığı (°C)")
    ax1.plot(history["target"], color='black', linestyle='--', alpha=0.6, label="RL Setpoint (Hedef)")
    ax1.set_ylabel("Sıcaklık Değerleri")
    ax1.set_title("Döner Fırın Termal Analiz Dashboard", fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="lower right")

    # Yakıt Grafiği
    ax2.step(range(100), history["fuel"], color='royalblue', linewidth=2, label="MPC Yakıt Girişi (kg/s)")
    ax2.set_ylabel("Yakıt Akışı")
    ax2.set_xlabel("Zaman (Simülasyon Adımı)") # Eksik etiket eklendi
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    print("\nGrafik Hazırlanıyor. Lütfen pencereyi kontrol edin.")
    plt.show()
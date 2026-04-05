import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ml.rl_agent import RotaryKilnEnv # RL ortamımız

def run_industrial_simulation(model_path=None):
    # 1. Ortamı Başlat
    env = RotaryKilnEnv()
    
    # 2. Modeli Yükle veya Eğit
    if model_path:
        print(f"--- Mevcut model yükleniyor: {model_path} ---")
        model = PPO.load(model_path)
    else:
        print("--- Sıfırdan eğitim başlıyor (MCP + RL Hibrit) ---")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=5000)
        model.save("models/rotary_kiln_v1")

    # 3. Test ve Görselleştirme (Simülasyon)
    obs, _ = env.reset()
    history = {"temp": [], "fuel": [], "target": []}
    
    print("\n--- Simülasyon Çalışıyor (100 Adım) ---")
    for i in range(100):
        # RL Ajanı karar verir (Hangi setpoint?)
        action, _states = model.predict(obs, deterministic=True)
        
        # Ortamda bir adım ilerle (İçeride MPC çalışır)
        obs, reward, done, truncated, info = env.step(action)
        
        # Verileri kaydet
        history["temp"].append(obs[0])   # Mevcut Sıcaklık
        history["fuel"].append(obs[1])   # MPC'nin bastığı yakıt
        history["target"].append(1400 + action[0]) # RL'nin belirlediği hedef
        
    # 4. Grafiklerle Analiz
    plot_results(history)

def plot_results(history):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Sıcaklık Grafiği
    ax1.set_xlabel('Zaman (saniye)')
    ax1.set_ylabel('Sıcaklık (°C)', color='tab:red')
    ax1.plot(history["temp"], label="Fırın Sıcaklığı", color='tab:red', linewidth=2)
    ax1.plot(history["target"], '--', label="RL Hedef (Setpoint)", color='black', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)

    # Yakıt Grafiği (İkinci Y ekseni)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Yakıt Akışı (kg/s)', color='tab:blue')
    ax2.step(range(len(history["fuel"])), history["fuel"], label="MPC Yakıt Girişi", color='tab:blue', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Endüstriyel AI: MCP + RL Hibrit Kontrol Performansı')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_industrial_simulation()
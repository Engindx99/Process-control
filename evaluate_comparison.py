import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. SİMÜLASYON AYARLARI VE BOZUCU ETKİ FONKSİYONU ---
def run_comparison_simulation():
    print("🔄 Simülasyon başlatılıyor (Bozucu etkiler dahil)...")
    steps = 3000
    setpoint = 1450.0
    
    # Veri saklama listeleri
    mpc_data = []
    hybrid_data = []

    # Başlangıç durumları
    temp_mpc = 1400.0
    temp_hybrid = 1400.0

    for i in range(steps):
        # --- BOZUCU ETKİLER (DISTURBANCES) ---
        disturbance = 0
        if i == 1000:
            disturbance = -40.0  # Ani soğuk malzeme girişi
            print(f"⚠️ Adım {i}: Soğuk malzeme girişi (-40°C)")
        elif i == 2000:
            disturbance = -25.0  # Yakıt kalitesi dalgalanması
            print(f"⚠️ Adım {i}: Yakıt kalitesi düştü (-25°C)")

        # Sıcaklıkları bozucu etkiyle güncelle
        temp_mpc += disturbance
        temp_hybrid += disturbance

        # --- MODEL TEPKİLERİ (TEMSİLİ HESAPLAMA) ---
        # Not: Burada gerçek MPC ve RL objelerin varsa onları çağırmalısın.
        # Bu kısım grafik yapısını simüle eder.
        
        # Pure MPC: Daha sert ve salınımlı tepki
        error_mpc = setpoint - temp_mpc
        temp_mpc += 0.05 * error_mpc + np.random.normal(0, 0.1) 
        
        # Hybrid: Daha pürüzsüz ve hızlı sönümlenen tepki
        error_hybrid = setpoint - temp_hybrid
        temp_hybrid += 0.08 * error_hybrid + np.random.normal(0, 0.05)

        # Verileri kaydet
        mpc_data.append({"step": i, "temperature": temp_mpc, "error": temp_mpc - setpoint})
        hybrid_data.append({"step": i, "temperature": temp_hybrid, "error": temp_hybrid - setpoint})

    # CSV olarak kaydet
    if not os.path.exists('data'): os.makedirs('data')
    pd.DataFrame(mpc_data).to_csv("data/pure_mpc_results.csv", index=False)
    pd.DataFrame(hybrid_data).to_csv("data/hybrid_rl_results.csv", index=False)
    print("✅ Simülasyon tamamlandı ve veriler kaydedildi.")

# --- 2. GÖRSELLEŞTİRME FONKSİYONU ---
def plot_results():
    print("📊 Grafikler oluşturuluyor...")
    df_mpc = pd.read_csv("data/pure_mpc_results.csv")
    df_hybrid = pd.read_csv("data/hybrid_rl_results.csv")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Üst Grafik: Sıcaklık
    axes[0].plot(df_mpc['step'], df_mpc['temperature'], label='Pure MPC', color='tab:red', alpha=0.4, linestyle='--')
    axes[0].plot(df_hybrid['step'], df_hybrid['temperature'], label='Hybrid (MPC+RL)', color='tab:blue', lw=2)
    axes[0].axhline(y=1450, color='black', linestyle=':', label='Hedef (1450°C)')
    
    # Bozucu etki okları
    axes[0].annotate('Soğuk Malzeme', xy=(1000, 1410), xytext=(1100, 1380),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    axes[0].annotate('Düşük Yakıt', xy=(2000, 1425), xytext=(2100, 1400),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    axes[0].set_title("Döner Fırın Kontrolü: Bozucu Etki Altında Performans", fontsize=16)
    axes[0].set_ylabel("Sıcaklık (°C)")
    axes[0].legend()

    # Alt Grafik: Hata
    axes[1].fill_between(df_mpc['step'], df_mpc['error'], color='tab:red', alpha=0.1)
    axes[1].plot(df_mpc['step'], df_mpc['error'], color='tab:red', alpha=0.3, label='MPC Hatası')
    axes[1].plot(df_hybrid['step'], df_hybrid['error'], color='tab:blue', alpha=0.8, label='Hybrid Hatası')
    axes[1].set_ylabel("Hata (°C)")
    axes[1].set_xlabel("Adım (Step)")
    axes[1].legend()

    # Pylance 'rect' hatasını çözen tuple kullanımı
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

# --- ANA ÇALIŞTIRICI ---
if __name__ == "__main__":
    run_comparison_simulation()
    plot_results()
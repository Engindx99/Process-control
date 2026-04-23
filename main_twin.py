import yaml
import pandas as pd
import numpy as np
import logging
from src.burning_zone.burning_zone import RotaryKilnPlant
from src.mpc.mpc import MPC 
from src.filters.kalman import SelectiveKalmanFilter

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_twin_engine():
    # 1. En Optimize Edilmiş Config'i Yükle
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info("✅ Config dosyası başarıyla yüklendi.")
    except FileNotFoundError:
        logging.error("❌ config.yaml bulunamadı!")
        return

    # 2. Bileşenleri Başlat
    # seed=42 vererek Optuna sonuçlarıyla birebir aynı başlangıcı garanti ediyoruz
    plant = RotaryKilnPlant(seed=42) 
    obs = plant.reset() 
    mpc = MPC(config) 
    kf = SelectiveKalmanFilter(config) 
    
    # Zaman ve Hedef Tanımları
    TOTAL_SECONDS = 21600   # 6 Saatlik simülasyon
    CONTROL_INTERVAL = 10   # 10 saniyede bir MPC kararı (Endüstriyel standart)
    SETPOINT = config['system']['setpoint']
    
    # Başlangıç komutlarını config'deki initial değerlerden al
    fuel_cmd = config['mpc']['initial_fuel']
    fan_cmd = config['mpc']['initial_fan']
    history = []

    print(f"\n" + "="*50)
    print(f"🚀 DİJİTAL İKİZ MOTORU (v10.6) BAŞLATILDI")
    print(f"🎯 Hedef Sıcaklık: {SETPOINT}°C | O2: {config['system']['target_o2']}%")
    print(f"⏱️ Toplam Süre: {TOTAL_SECONDS} saniye")
    print("="*50 + "\n")

    for t in range(TOTAL_SECONDS):
        # --- KALMAN GÜNCELLEME (Sensör Filtreleme) ---
        # Basınç ve O2 gürültüsünü temizlemek için her saniye çalışır
        kf.update(obs['o2'], obs['pressure'])
        
        # --- MPC KONTROL DÖNGÜSÜ (Karar Mekanizması) ---
        if t % CONTROL_INTERVAL == 0:
            try:
                # Plant.history_fuel gönderilerek 10s gecikme (dead-time) MPC'ye bildirilir
                fuel_cmd, fan_cmd = mpc.get_action(
                    current_temp=obs['temp'], 
                    current_o2=obs['o2'], 
                    fuel_history=plant.history_fuel
                )
            except Exception as e:
                logging.error(f"MPC Hatası (Saniye {t}): {e}")
                break
        
        # --- FİZİKSEL ADIM (Fırın Tepkisi) ---
        obs = plant.step(fuel_cmd, fan_cmd)
        
        # --- VERİ KAYDI ---
        current_error = abs(obs['temp'] - SETPOINT)
        history.append({
            'second': t,
            'temp': obs['temp'],
            'o2': obs['o2'],
            'fuel': fuel_cmd,
            'fan': fan_cmd,
            'error': current_error,
            'pressure': obs['pressure']
        })

        # --- SAATLİK RAPORLAMA ---
        if t % 3600 == 0 and t > 0:
            cum_mae = np.mean([h['error'] for h in history])
            recent_mae = np.mean([h['error'] for h in history[-600:]]) # Son 10 dk
            
            print(f"📅 Saat {t//3600}: Sıcaklık={obs['temp']:.2f}°C | "
                  f"Kümülatif MAE={cum_mae:.4f} | "
                  f"Son 10dk MAE={recent_mae:.4f}")

    # 3. Sonuçları CSV Olarak Kaydet
    df = pd.DataFrame(history)
    output_file = "kiln_6h_final_results.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print(f"🏆 SİMÜLASYON BAŞARIYLA TAMAMLANDI")
    print(f"📊 Final Genel MAE: {df['error'].mean():.4f}")
    print(f"📈 Son 1 Saat MAE: {df['error'].tail(3600).mean():.4f}")
    print(f"💾 Sonuçlar '{output_file}' dosyasına yazıldı.")
    print("="*50)

if __name__ == "__main__":
    run_twin_engine()
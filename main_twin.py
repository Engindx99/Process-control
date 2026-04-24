import yaml
import pandas as pd
import numpy as np
import logging
import copy
from Rotaryklin.Rotary_klin import RotaryKilnPlant
from src.mpc.mpc import MPC 
from src.filters.kalman import KalmanFilter

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_twin_engine():
    # 1. Optimize Edilmiş Config'i Yükle
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info("✅ v10.11 Master Config başarıyla yüklendi.")
    except Exception as e:
        logging.error(f"❌ Config yükleme hatası: {e}")
        return

    # 2. Bileşenleri Başlat
    plant = RotaryKilnPlant(seed=42) 
    obs = plant.reset() 
    mpc = MPC(config) 
    kf = KalmanFilter(config) 
    
    # Simülasyon Ayarları
    TOTAL_SECONDS = 21600   # 6 Saat
    CONTROL_INTERVAL = 10   # 10s MPC adımı
    SETPOINT = config['system']['setpoint']
    TARGET_O2 = config['system']['target_o2']
    
    # Başlangıç Değerleri
    fuel_cmd = config['mpc'].get('initial_fuel', 18.0)
    fan_cmd = config['mpc'].get('initial_fan', 900.0)
    
    # DÜZELTME: history_fuel hatası için manuel liste yönetimi
    # MPC 10 adımlık (gecikme süresi) bir yakıt geçmişi bekliyor
    manual_fuel_history = [fuel_cmd] * 10 
    
    history = []

    print(f"\n" + "="*60)
    print(f"🚀 DİJİTAL İKİZ MOTORU (v10.12 FIX) BAŞLATILDI")
    print(f"🎯 Hedef: {SETPOINT}°C | O2: {TARGET_O2}%")
    print("="*60 + "\n")

    for t in range(TOTAL_SECONDS):
        # --- 1. KALMAN FİLTRELEME ---
        filt_o2, filt_press = kf.filter(obs['o2'], obs['pressure'])
        
        # --- 2. MPC KONTROL DÖNGÜSÜ ---
        if t % CONTROL_INTERVAL == 0:
            try:
                # DÜZELTME: plant.history_fuel yerine manuel liste kullanıldı
                fuel_cmd, fan_cmd = mpc.get_action(
                    current_temp=obs['temp'], 
                    current_o2=filt_o2, 
                    current_press=filt_press, 
                    fuel_history=manual_fuel_history
                )
            except Exception as e:
                logging.error(f"❌ MPC Çözücü Hatası (Saniye {t}): {e}")
                break # Hata olursa döngüden çık
        
        # --- 3. FİZİKSEL ADIM ---
        obs = plant.step(fuel_cmd, fan_cmd)
        
        # Yakıt geçmişini güncelle (MPC'nin beklediği lag listesi)
        manual_fuel_history.append(obs['fuel'])
        manual_fuel_history.pop(0)
        
        # --- 4. VERİ KAYDI ---
        current_error = abs(obs['temp'] - SETPOINT)
        history.append({
            'second': t,
            'temp': obs['temp'],
            'o2': obs['o2'],
            'filt_o2': filt_o2,
            'pressure': obs['pressure'],
            'filt_press': filt_press,
            'fuel': fuel_cmd,
            'fan': fan_cmd,
            'error': current_error
        })

        if t % 3600 == 0 and t > 0:
            avg_err = np.mean([h['error'] for h in history[-3600:]])
            print(f"📅 Saat {t//3600} | Temp: {obs['temp']:.2f}°C | Son 1 Saat MAE: {avg_err:.4f}")

    # 5. Sonuçları Kaydet (Hata Kontrollü)
    if not history:
        logging.error("❌ Hiç veri toplanamadı. Simülasyon başlamadan çöktü.")
        return

    df = pd.DataFrame(history)
    output_file = "kiln_v10_12_results.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print(f"🏆 SİMÜLASYON TAMAMLANDI | Final MAE: {df['error'].mean():.4f}")
    print(f"💾 Sonuçlar '{output_file}' dosyasına kaydedildi.")
    print("="*60)

if __name__ == "__main__":
    run_twin_engine()
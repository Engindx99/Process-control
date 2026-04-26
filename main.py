import pandas as pd
import numpy as np
import traceback
from simulation.config import KilnConfig
from models.kiln import KilnModel
from utils.visualizer import KilnVisualizer
from control.casadi_mpc import KilnCasadiMPC

def main():
    # 1. Ayarları YAML'dan Yükle
    try:
        cfg = KilnConfig.from_yaml("config.yaml")
        print(f"--- Yapılandırma Yüklendi: dt={cfg.dt}s, Toplam Adım={cfg.simulation_steps} ---")
    except Exception as e:
        print(f"Yapılandırma dosyası okunamadı: {e}")
        return

    # 2. Bileşenleri Başlat
    model = KilnModel(cfg)
    mpc = KilnCasadiMPC(cfg)
    viz = KilnVisualizer(cfg)
    
    history = []
    save_interval = 2000  # Log ekranına basma sıklığı

    print(f"--- Dijital İkiz Operasyonu Başlıyor ---")
    
    try:
        for step in range(cfg.simulation_steps):
            
            # 3. MPC Kontrol Döngüsü
            if step % cfg.control_interval == 0:
                # Durum vektörünü malzeme çıkışından (fırın sonu) alıyoruz
                state_vec = [
                    model.state.T_solid[-1], # Katı çıkış sıcaklığı
                    model.state.c_caco3[-1], # Kalan kireçtaşı
                    model.state.c_cao[-1],   # Serbest kireç
                    model.state.c_c2s[-1],   # Belit
                    model.state.c_c3s[-1]    # Alit (Hedef Ürün)
                ]
                
                # MPC'den yeni yakıt sıcaklığını al
                try:
                    optimized_fuel_temp = mpc.compute_control(state_vec)
                    
                    # KRİTİK DÜZELTME: 
                    # Yakıt (Gaz) fırına brülör ucundan girer. 
                    # Ters akışlı modelde bu genellikle 0. indekstir (Giriş).
                    # Eğer T_gas[-1] ürün çıkışıyla aynı yerdeyse orayı da güncelleyebilirsin.
                    model.state.T_gas[0] = optimized_fuel_temp 
                    
                except Exception as mpc_err:
                    print(f"MPC Hatası (Adım {step}): {mpc_err}")

            # 4. Fiziksel Simülasyon Adımı (Fırın bir adım ilerler)
            model.step()
            
            # 5. Veri Kaydı ve Terminal Bilgilendirme
            if step % save_interval == 0:
                s = model.state
                # Kalsinasyon yüzdesi: (1 - mevcut_caco3)
                calc_pct = (1.0 - s.c_caco3[-1]) * 100
                c3s_pct = s.c_c3s[-1] * 100
                
                # Alev sıcaklığı olarak brülör ucunu (girişi) takip ediyoruz
                current_flame = s.T_gas[0]
                
                print(f"Zaman: {model.time:>7.1f}s | T_çıkış: {s.T_solid[-1]:>6.1f}K | "
                      f"Kals: %{calc_pct:>5.2f} | C3S: %{c3s_pct:>5.2f} | "
                      f"Alev: {current_flame:>6.1f}K")
                
                history.append({
                    'time': model.time,
                    't_solid_exit': s.T_solid[-1],
                    'c3s_pct': c3s_pct,
                    'fuel_temp': current_flame
                })

        # 6. Simülasyon Sonu: Kayıt ve Görselleştirme
        print("\nSimülasyon başarıyla tamamlandı.")
        df = pd.DataFrame(history)
        df.to_csv("data/outputs/kiln_final_results.csv", index=False)
        
        print("Final profili oluşturuluyor...")
        viz.plot_state(model.state, model.time)

    except KeyboardInterrupt:
        print("\nSimülasyon kullanıcı tarafından durduruldu. Grafik hazırlanıyor...")
        viz.plot_state(model.state, model.time)
    except Exception as e:
        print(f"\nBeklenmedik Hata: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
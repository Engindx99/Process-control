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
    # Log ekranına basma sıklığı (Görünürlüğü artırmak için biraz düşürülebilir)
    save_interval = 500  

    print(f"--- Dijital İkiz Operasyonu Başlıyor (Hedef: C3S Üretimi) ---")
    
    try:
        for step in range(cfg.simulation_steps):
            
            # 3. MPC Kontrol Döngüsü
            if step % cfg.control_interval == 0:
                # Durum vektörünü malzeme çıkışından (fırın sonu) alıyoruz
                # x_sym sıralamasıyla uyumlu: [T_s, CaCO3, CaO, C2S, C3S]
                state_vec = [
                    model.state.T_solid[-1], 
                    model.state.c_caco3[-1], 
                    model.state.c_cao[-1],   
                    model.state.c_c2s[-1],   
                    model.state.c_c3s[-1]    
                ]
                
                try:
                    # MPC'den yeni yakıt sıcaklığını al
                    optimized_fuel_temp = mpc.compute_control(state_vec)
                    
                    # KRİTİK: Yakıtı brülör ucuna (çıkış tarafı) enjekte ediyoruz
                    model.state.T_gas[-1] = optimized_fuel_temp 
                    
                except Exception as mpc_err:
                    print(f"\nMPC Hatası (Adım {step}): {mpc_err}")

            # 4. Fiziksel Simülasyon Adımı
            model.step()
            
            # 5. Veri Kaydı ve Terminal Bilgilendirme
            if step % save_interval == 0:
                s = model.state
                calc_pct = (1.0 - s.c_caco3[-1]) * 100
                
                # HASSASİYET GÜNCELLEMESİ: 
                # C3S başlangıçta çok küçük değerler aldığı için .6f hassasiyet ekledik.
                c3s_val = s.c_c3s[-1] * 100
                current_flame = s.T_gas[-1]
                
                print(f"Zaman: {model.time:>7.1f}s | T_çıkış: {s.T_solid[-1]:>6.1f}K | "
                      f"Kals: %{calc_pct:>6.2f} | C3S: %{c3s_val:>8.6f} | "
                      f"Alev: {current_flame:>6.1f}K")
                
                history.append({
                    'time': model.time,
                    't_solid_exit': s.T_solid[-1],
                    'c_caco3': s.c_caco3[-1],
                    'c_c3s': s.c_c3s[-1],
                    'fuel_temp': current_flame
                })

        # 6. Simülasyon Sonu
        print("\nSimülasyon başarıyla tamamlandı.")
        df = pd.DataFrame(history)
        df.to_csv("data/outputs/kiln_final_results.csv", index=False)
        
        print("Final grafikleri hazırlanıyor...")
        viz.plot_state(model.state, model.time)

    except KeyboardInterrupt:
        print("\nSimülasyon durduruldu. Mevcut veriler kaydediliyor...")
        viz.plot_state(model.state, model.time)
    except Exception as e:
        print(f"\nBeklenmedik Hata: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
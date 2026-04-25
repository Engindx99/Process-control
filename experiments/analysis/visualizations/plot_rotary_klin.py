import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.Rotaryklin.Rotary_klin import Kiln1D, SIGMA

def visualize_kiln_advanced(kiln, history):
    """
    Non-lineer etkileri (ısı sıçraması, kalsinasyon hızı) ön plana çıkaran 
    gelişmiş fırın dashboard'u.
    """
    # Veri Hazırlığı
    steps = np.arange(len(history))
    t_burn = np.array([h["T_burning"] for h in history])
    x_mean = np.array([h["X_mean"] for h in history])
    o2_exit = np.array([h["O2_out"] for h in history])
    
    x_axis = np.linspace(0, kiln.L, kiln.N)
    
    # Isı Akışı ve Gradyan Analizi (Doğrusallığı görmek için)
    dT_dx = np.gradient(kiln.Ts - 273) # Malzeme sıcaklık değişim hızı

    # Plot Ayarları
    plt.style.use('dark_background') # Daha profesyonel görünüm için
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"Rotary Kiln Digital Twin - Steady State Analysis", fontsize=18, y=0.95)
    gs = GridSpec(3, 3, figure=fig)

    # 1. ANA PROFİL: SICAKLIK (Geniş Üst Panel)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x_axis, kiln.Tg - 273, color='#3498db', label='Gaz (Alev)', linewidth=1.5, alpha=0.6)
    ax1.plot(x_axis, kiln.Ts - 273, color='#f39c12', label='Malzeme (Klinker)', linewidth=3)
    ax1.fill_between(x_axis, kiln.Ts - 273, color='#f39c12', alpha=0.1)
    ax1.set_title("Fırın Boyu Sıcaklık Dağılımı ve Isıl Sıçrama Bölgesi", fontsize=14)
    ax1.set_ylabel("Sıcaklık (°C)")
    ax1.grid(True, alpha=0.2)
    ax1.legend()

    # 2. ISIL KAZANÇ HIZI (Doğrusallığın kırıldığı yer)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(x_axis, dT_dx, color='#e74c3c', alpha=0.3)
    ax2.plot(x_axis, dT_dx, color='#e74c3c', linewidth=2)
    ax2.set_title("Isıl Kazanç Gradyanı (dT/dx)")
    ax2.set_ylabel("Sıcaklık Artış Hızı")
    ax2.grid(True, alpha=0.2)

    # 3. KALSİNASYON VE CO2 (Reaksiyon Takibi)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x_axis, kiln.X_calc, color='#9b59b6', linewidth=2.5, label='Kalsinasyon %')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x_axis, kiln.CO2 * 100, color='#1abc9c', linestyle='--', label='CO2 %')
    ax3.set_title("Reaksiyon İlerlemesi")
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc='upper left')

    # 4. RADYASYON AKISI (Enerji Transferi)
    ax4 = fig.add_subplot(gs[1, 2])
    q_rad = (SIGMA * kiln.eps * (kiln.Tg**4 - kiln.Ts**4)) / 1000
    ax4.bar(x_axis, q_rad, width=0.8, color='#f1c40f', alpha=0.5)
    ax4.set_title("Radyasyon Enerji Akısı (kW/m²)")
    ax4.grid(True, alpha=0.2)

    # 5. ZAMANSAL GELİŞİM: BURNING ZONE
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(steps, t_burn, color='#e67e22')
    ax5.axhline(1450, color='red', linestyle='--')
    ax5.set_title("Burning Zone Sıcaklık Evrimi")
    ax5.set_xlabel("Adım")
    ax5.set_ylabel("°C")

    # 6. ZAMANSAL GELİŞİM: KALSİNASYON DERECESİ
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(steps, x_mean, color='#9b59b6')
    ax6.set_title("Ortalama Kalsinasyon (Zaman)")
    ax6.set_ylim(0, 1)

    # 7. O2 ÇIKIŞI (Emisyon Takibi)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(steps, o2_exit, color='#2ecc71')
    ax7.set_title("Çıkış O2 Kararlılığı")
    ax7.set_ylabel("Mole Oranı")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =========================================================
# ÇALIŞTIRMA BLOĞU
# =========================================================
if __name__ == "__main__":
    # N değerini çok yüksek tutma, dx büyüdükçe kararlılık artar
    kiln = Kiln1D(N=60, L=60) 
    history = []

    print("Simülasyon başlıyor...")
    # Patlamayı önlemek için ilk 1000 adımda yakıtı sabit tutalım (Isınma evresi)
    for t in range(5000):
        if t < 1000:
            kiln.fuel = 34.0
        else:
            # Salınımı çok yavaşlat (0.01 yerine 0.002)
            kiln.fuel = 34 + 1.5 * np.sin(t * 0.002)
        
        # dt değerini 0.001'e çekmek patlamayı %99 durdurur
        res = kiln.step(dt=0.001) 
        history.append(res)

        # Hata kontrolü: Eğer sıcaklık saçmalarsa simülasyonu durdur
        if np.isnan(res['T_burning']) or res['T_burning'] > 5000:
            print(f"Sayısal patlama oluştu! Adım: {t}")
            break

    print(f"Simülasyon bitti. Final T_burn: {history[-1]['T_burning']:.2f} °C")
    visualize_kiln_advanced(kiln, history)
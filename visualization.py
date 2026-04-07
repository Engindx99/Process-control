import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_dashboard():
    # Dosya yolunu kontrol et
    csv_path = "outputs/simulasyon_verileri.csv"
    
    if not os.path.exists(csv_path):
        print(f"\nHata: {csv_path} bulunamadı! Lütfen önce main.py'yi çalıştırın.")
        return

    try:
        # Veriyi Oku
        df = pd.read_csv(csv_path)
        
        # Dashboard Düzeni (2 satır, 2 sütun)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.2)

        # 1. Grafik: Sıcaklık ve Hedef
        ax1.plot(df["adim"], df["sicaklik"], color='crimson', label="Fırın Sıcaklığı (°C)", linewidth=2)
        ax1.plot(df["adim"], df["hedef"], color='black', linestyle='--', alpha=0.7, label="RL Hedef (Setpoint)")
        ax1.set_title("Termal Kontrol Performansı", fontweight='bold')
        ax1.set_ylabel("Sıcaklık (°C)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Grafik: Oksijen Seviyesi
        ax2.plot(df["adim"], df["oksijen"], color='forestgreen', label="O2 Seviyesi (%)", linewidth=2)
        ax2.axhline(y=4.0, color='red', linestyle=':', label="İdeal O2 (%4)")
        ax2.set_title("Yanma Verimliliği (Oksijen Analizi)", fontweight='bold')
        ax2.set_ylabel("O2 %")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Grafik: Yakıt Girişi
        ax3.step(df["adim"], df["yakit"], color='royalblue', label="Yakıt Akışı (kg/s)")
        ax3.set_title("Enerji Tüketimi (Yakıt)", fontweight='bold')
        ax3.set_ylabel("Debi")
        ax3.set_xlabel("Simülasyon Adımı")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Grafik: Fan Hızı (Hava)
        ax4.step(df["adim"], df["fan"], color='darkorange', label="Fan Hızı (RPM/Hava)")
        ax4.set_title("Hava Besleme ve Soğutma Etkisi", fontweight='bold')
        ax4.set_ylabel("Hız/Debi")
        ax4.set_xlabel("Simülasyon Adımı")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.suptitle("Gelişmiş Döner Fırın Dijital İkiz Analiz Dashboard", fontsize=16, fontweight='bold')
        
        # Kaydet ve Göster
        plot_path = "outputs/analiz_raporu.png"
        plt.savefig(plot_path, dpi=300)
        print(f"\n--- Analiz raporu oluşturuldu: {plot_path} ---")
        plt.show()

    except Exception as e:
        print(f"Grafik çizilirken hata oluştu: {e}")

if __name__ == "__main__":
    generate_dashboard()
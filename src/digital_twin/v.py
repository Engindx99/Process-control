import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_kiln_results(csv_file="kiln_dataset.csv"):
    if not os.path.exists(csv_file):
        print(f"❌ Hata: {csv_file} bulunamadı. Önce simülasyonu çalıştırın.")
        return

    # Veriyi yükle
    df = pd.read_csv(csv_file)
    
    # Görselleştirme ayarları
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Döner Fırın Dijital İkiz Analizi ({len(df)} Adım)", fontsize=20)

    # 1. Panel: Sıcaklık ve Hedef
    axes[0, 0].plot(df['adim'], df['sicaklik'], color='darkred', label='Fırın Sıcaklığı')
    axes[0, 0].axhline(y=1450, color='gray', linestyle='--', label='İdeal Hedef (1450°C)')
    axes[0, 0].set_title("Sıcaklık Zaman Serisi")
    axes[0, 0].set_ylabel("Derece (°C)")
    axes[0, 0].legend()

    # 2. Panel: Yakıt ve O2 İlişkisi (Çift Eksen)
    ax2 = axes[0, 1].twinx()
    axes[0, 1].plot(df['adim'][:500], df['fuel'][:500], color='orange', label='Yakıt', alpha=0.8)
    ax2.plot(df['adim'][:500], df['o2'][:500], color='blue', label='O2', alpha=0.6)
    axes[0, 1].set_title("Yakıt vs O2 Dengesi (İlk 500 Adım)")
    axes[0, 1].set_ylabel("Yakıt Mik. (m³/h)")
    ax2.set_ylabel("O2 Oranı (%)")
    # Legend birleştirme
    lines, labels = axes[0, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # 3. Panel: Verimlilik Dağılımı
    sns.histplot(df['efficiency'], kde=True, ax=axes[1, 0], color='green')
    axes[1, 0].set_title("Yanma Verimliliği Dağılımı")
    axes[1, 0].set_xlabel("Verim (0.0 - 1.0)")

    # 4. Panel: Isı Haritası (Korelasyon)
    corr = df[['fuel', 'fan', 'sicaklik', 'o2', 'efficiency']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title("Değişkenler Arası Korelasyon")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Kaydet ve Göster
    output_plot = "experiments/plots/twin_analysis.png"
    os.makedirs("experiments/plots", exist_ok=True)
    plt.savefig(output_plot)
    print(f"✅ Görselleştirme '{output_plot}' adresine kaydedildi.")
    plt.show()

if __name__ == "__main__":
    visualize_kiln_results()
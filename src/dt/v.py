import os
import pandas as pd
import matplotlib.pyplot as plt


def analyze_kiln_results():

    # 📌 bulunduğun klasör = src/dt
    base_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(base_dir, "kiln_dataset.csv")

    print("📁 Okunan dosya:", file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ {file_path} bulunamadı. Önce simülasyonu çalıştırın."
        )

    df = pd.read_csv(file_path)

    plt.figure(figsize=(14, 6))
    plt.plot(df["adim"], df["sicaklik"], label="Temp")
    plt.axhline(1450, linestyle="--")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    analyze_kiln_results()
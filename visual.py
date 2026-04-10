import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from digital import RotaryKilnDigitalTwin

df = pd.read_csv("firin_dataset_5k.csv")

def plot_1(df):
    plt.figure()
    plt.scatter(df["fuel"], df["sicaklik"], c=df["o2"], cmap="viridis")
    plt.colorbar(label="O2")
    plt.xlabel("Fuel")
    plt.ylabel("Sıcaklık (°C)")
    plt.title("Yakıt - Sıcaklık İlişkisi (O2 etkili)")
    plt.grid()
    plt.show()


def plot_2(df):
    plt.figure()
    sns.regplot(x="fuel", y="o2", data=df, scatter_kws={"alpha":0.5})
    plt.xlabel("Fuel")
    plt.ylabel("O2 %")
    plt.title("Yakıt - Oksijen İlişkisi")
    plt.grid()
    plt.show()


def plot_3(df):
    plt.figure()
    plt.scatter(df["o2"], df["sicaklik"], c="green", alpha=0.6)
    plt.xlabel("O2 %")
    plt.ylabel("Sıcaklık (°C)")
    plt.title("Oksijen - Sıcaklık İlişkisi")
    plt.grid()
    plt.show()


def plot_4(df):
    plt.figure()
    corr = df[["fuel", "fan", "sicaklik", "o2"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Değişken Etkileşim Matrisi")
    plt.show()


# ---- RUN SEQUENCE ----
plot_1(df)
plot_2(df)
plot_3(df)
plot_4(df)
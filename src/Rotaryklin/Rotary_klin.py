import numpy as np

SIGMA = 5.67e-8
R = 8.314

class Kiln1D:
    def __init__(self, N=60, L=60):
        self.N, self.L = N, L
        self.dx = L / N
        self.eps = 0.95 # Emisiviteyi biraz artırdık (radyasyon transferi için)
        self.h = 220    # Konveksiyon katsayısı (150-250 arası idealdir)

        # Başlangıç Koşulları
        self.Ts = np.linspace(250 + 273, 1300 + 273, N)
        self.Tg = np.linspace(800 + 273, 2000 + 273, N)
        self.O2 = np.ones(N) * 0.21
        self.CO2 = np.ones(N) * 0.05
        self.X_calc = np.linspace(0.0, 1.0, N)

        # Fiziksel Parametreler
        self.u_s, self.u_g = 0.012, 1.8
        self.rho_s, self.rho_g = 1500, 1.1
        self.Cp_s, self.Cp_g = 1150, 1250
        self.fuel = 36.5 # 1450C hedefi için optimize yakıt miktarı

    def step(self, dt=0.002):
        # Önceki değerleri sakla
        Ts_old, Tg_old, X_old = self.Ts.copy(), self.Tg.copy(), self.X_calc.copy()

        # 1. TAŞINIM (Upwind)
        self.Ts[1:] -= (self.u_s * dt / self.dx) * (Ts_old[1:] - Ts_old[:-1])
        self.Tg[:-1] += (self.u_g * dt / self.dx) * (Tg_old[1:] - Tg_old[:-1])

        # 2. ISI TRANSFERİ (Radyasyon + Konveksiyon)
        # Delta T'yi kapatmak için transferi daha efektif hesaplıyoruz
        q_rad = SIGMA * self.eps * (Tg_old**4 - Ts_old**4)
        q_conv = self.h * (Tg_old - Ts_old)
        
        # 3. REAKSİYONLAR
        # Kalsinasyon hızı (Arrhenius)
        r_calc = 1.2e5 * np.exp(-145000 / (R * np.clip(Ts_old, 300, 2500))) * (1 - X_old)
        dX = np.clip(r_calc * dt, 0, 0.01)
        self.X_calc = np.clip(X_old + dX, 0, 1)

        # Yanma (Alev profili - Daha yayvan ve kararlı)
        x_coords = np.linspace(0, self.L, self.N)
        flame = np.exp(-((x_coords - 55)**2) / 45)
        r_comb = (self.fuel * 0.008) * flame * dt

        # 4. ENERJİ DENGESİ
        # Isı Kayıpları (Fırın kabuğu kaybı - 1500C üzerinde denge kurar)
        q_loss = 18.0 * (Ts_old - 320) 

        # Malzeme ısınma ataleti (Kalsinasyon biterken ısınma hızlanır)
        material_inertia = self.rho_s * self.Cp_s * (0.12 + 0.08 * (1 - self.X_calc))
        gas_inertia = self.rho_g * self.Cp_g

        dT_s = (q_conv + q_rad - (r_calc * 175000) - q_loss) * dt / material_inertia
        dT_g = (-q_conv - q_rad + (r_comb * 4.2e7)) * dt / gas_inertia

        # 5. GÜNCELLEME VE LİMİTLEME (Zikzakları önler)
        self.Ts += np.clip(dT_s, -5, 5)
        self.Tg += np.clip(dT_g, -8, 8)
        
        # Gaz Bileşimi (Zikzakları önlemek için katsayıyı düşürdük)
        self.O2 = np.clip(self.O2 - r_comb * 0.1, 0.01, 0.21)
        self.CO2 = np.clip(self.CO2 + r_comb * 0.1 + dX * 0.05, 0.02, 0.35)

        # 6. SINIR KOŞULLARI
        self.Ts[0] = 300 + 273
        self.Tg[-1] = 2050 + 273 
        self.O2[-1] = 0.21

        return {
            "T_burning": float(self.Ts[int(self.N * 0.95)] - 273),
            "O2_out": float(self.O2[0]),
            "CO2_out": float(self.CO2[0]),
            "X_mean": float(np.mean(self.X_calc))
        }
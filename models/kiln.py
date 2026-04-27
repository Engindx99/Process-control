import numpy as np
from simulation.solver import KilnSolver
from models.state import KilnState

class KilnModel:
    def __init__(self, config):
        self.cfg = config
        self.state = KilnState(config.n_zones)
        self.solver = KilnSolver(config)
        self.time = 0.0
        
        # İlk kararlı profil ataması (Warm Start)
        self._initialize_internal_state()

    def _initialize_internal_state(self):
        """Fırını buz gibi başlatmak yerine çalışma sıcaklığına yakın başlatır."""
        n = self.cfg.n_zones
        # Malzeme: Girişten çıkışa 400K -> 1400K lineer artış
        self.state.T_solid = np.linspace(400.0, 1400.0, n)
        # Gaz: Girişten brülöre 1000K -> 2200K lineer artış
        self.state.T_gas = np.linspace(1000.0, 2200.0, n)
        
        # Kimyasal profil: Girişte %100 CaCO3, çıkışa doğru azalır
        self.state.c_caco3 = np.linspace(1.0, 0.1, n)
        self.state.c_cao = np.linspace(0.0, 0.5, n)
        self.state.c_c3s = np.zeros(n)

    def apply_boundary_conditions(self):
        """
        Sınır şartlarını her adımda uygular. 
        DİKKAT: Sadece giriş (index 0) sabitlenir, iç bölgelerin ısınması engellenmez.
        """
        # 1. Hammadde Besleme (Z=0)
        # Sadece ilk hücreyi 1000K (farin sıcaklığı) yapıyoruz.
        self.state.T_solid[0] = 1000.0  
        self.state.c_caco3[0] = 1.0     
        
        # 2. Gaz Çıkışı ve Brülör (Z=L)
        # T_gas[-1] burada EZİLMEMELİ. O değer main.py'daki MPC'den geliyor.
        pass

    def step(self):
        """Simülasyonu bir adım ilerletir ve termal dengeyi korur."""
        # Sınır şartlarını (besleme) tazele
        self.apply_boundary_conditions()
        
        # MPC'den main.py aracılığıyla gelen brülör sıcaklığını yedekle
        current_burner_temp = self.state.T_gas[-1]
        
        try:
            # Diferansiyel denklem çözücü (Solver) adımını çalıştır
            T_s_new, T_g_new, C_new = self.solver.solve_step(self.state)
            
            # --- GÜNCELLEME VE KORUMA ---
            self.state.T_solid = T_s_new
            self.state.T_gas = T_g_new
            
            # KRİTİK: Solver'ın gazın en son hücresini (brülörü) 
            # hesaplamayla bozmasına izin verme, MPC'nin dediği kalsın.
            self.state.T_gas[-1] = current_burner_temp
            
            self.state.c_caco3 = C_new
            
            # Zamanı konfigürasyondaki dt kadar ilerlet
            self.time += self.cfg.dt
            
        except Exception as e:
            print(f"Solver hatası (Zaman {self.time}): {e}")
            raise
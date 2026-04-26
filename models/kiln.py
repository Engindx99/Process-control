from simulation.solver import KilnSolver
from models.state import KilnState

class KilnModel:
    def __init__(self, config):
        self.cfg = config
        self.state = KilnState(config.n_zones)
        self.solver = KilnSolver(config)
        self.time = 0.0
        
        # İlk değer ataması
        self._initialize_internal_state()

    def _initialize_internal_state(self):
        """Fırının başlangıç durumunu belirler."""
        # Malzeme profili: Giriş 1000K, fırın içi kademeli artış varsayımı
        self.state.T_solid[:] = 1000.0
        self.state.c_caco3[:] = 1.0
        
        # Gaz profili: Brülör ucu 1800K'dan başlar
        self.state.T_gas[:] = 1200.0
        self.state.T_gas[-1] = 1800.0

    def apply_boundary_conditions(self):
        """
        Sınır şartlarını her adımda uygular.
        NOT: T_gas[-1] burada sabitlenmemiştir, çünkü MPC tarafından yönetilir.
        """
        # 1. Hammadde Girişi (Z=0)
        # Endüstriyel gerçeklik: Ön ısıtıcı kulesinden gelen farin yaklaşık 1000K'dır.
        self.state.T_solid[0] = 1000.0  
        self.state.c_caco3[0] = 1.0     
        
        # 2. Gaz Çıkışı / Brülör Girişi (Z=L)
        # T_gas[-1] değerine dokunmuyoruz; o main.py içindeki MPC döngüsünden geliyor.
        pass

    def step(self):
        """Simülasyonu bir zaman adımı ilerletir."""
        # Sınır şartlarını koru (Giriş beslemesi ve saflık)
        self.apply_boundary_conditions()
        
        # Mevcut durumdaki T_gas[-1] (MPC'den gelen değer) ile diferansiyel denklemleri çöz
        try:
            T_s_new, T_g_new, C_new = self.solver.solve_step(self.state)
            
            # State güncelleme (Vektörel kopyalama)
            self.state.T_solid = T_s_new
            self.state.T_gas = T_g_new
            self.state.c_caco3 = C_new
            
            # Zamanı ilerlet
            self.time += self.cfg.dt
            
        except Exception as e:
            print(f"Solver hatası (Zaman {self.time}): {e}")
            raise
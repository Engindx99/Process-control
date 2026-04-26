import numpy as np

class KilnState:
    def __init__(self, n_zones):
        self.T_solid = np.full(n_zones, 700.0)
        self.T_gas = np.full(n_zones, 1200.0)
        self.c_caco3 = np.full(n_zones, 1.0)
        self.c_cao = np.zeros(n_zones)
        self.c_c2s = np.zeros(n_zones)
        self.c_c3s = np.zeros(n_zones)
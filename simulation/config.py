import yaml
from dataclasses import dataclass

@dataclass
class KilnConfig:
    # --- Geometri Parametreleri ---
    length: float
    n_zones: int
    
    # --- Fiziksel Parametreler ---
    dt: float
    velocity: float
    rho_solid: float
    cp_solid: float
    cp_gas: float
    
    # --- Simülasyon Ayarları ---
    simulation_steps: int
    control_interval: int
    
    # --- MPC / Kontrol Parametreleri ---
    prediction_horizon: int
    control_horizon: int
    prediction_step_size: float
    fuel_lb: float
    fuel_ub: float
    target_caco3: float
    target_c3s: float
    weight_calc: float
    weight_quality: float
    weight_smooth: float
    
    # --- Sabitler ---
    sigma: float = 5.67e-8
    R: float = 8.314

    @classmethod
    def from_yaml(cls, path="config.yaml"):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        # Hiyerarşik bloklar (Hala YAML içinde blok olanlar)
        geo = data.get('geometry', {})
        phys = data.get('physics', {})
        sim = data.get('simulation', {})
        
        # DİKKAT: Artık 'control' bloğu yok, veriler doğrudan 'data' içinde!
            
        return cls(
            # Geometri (Hiyerarşik)
            length=float(geo.get('length', 60.0)),
            n_zones=int(geo.get('n_zones', 60)),
            
            # Fizik (Hiyerarşik)
            dt=float(phys.get('dt', 0.1)),
            velocity=float(phys.get('velocity', 0.02)),
            rho_solid=float(phys.get('rho_solid', 1500.0)),
            cp_solid=float(phys.get('cp_solid', 1000.0)),
            cp_gas=float(phys.get('cp_gas', 1100.0)),
            
            # Simülasyon (Hiyerarşik)
            simulation_steps=int(sim.get('steps', 40000)),
            control_interval=int(sim.get('control_interval', 100)),
            
            # Kontrol (MPC) - Doğrudan ana sözlükten (data) okunuyor
            prediction_horizon=int(data.get('prediction_horizon', 100)),
            control_horizon=int(data.get('control_horizon', 10)),
            prediction_step_size=float(data.get('prediction_step_size', 40.0)),
            fuel_lb=float(data.get('fuel_lb', 2000.0)),
            fuel_ub=float(data.get('fuel_ub', 3000.0)),
            target_caco3=float(data.get('target_caco3', 0.02)),
            target_c3s=float(data.get('target_c3s', 0.60)),
            weight_calc=float(data.get('weight_calc', 5000.0)),
            weight_quality=float(data.get('weight_quality', 15000.0)),
            weight_smooth=float(data.get('weight_smooth', 100.0))
        )
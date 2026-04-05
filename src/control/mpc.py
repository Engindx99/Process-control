import numpy as np
from casadi import *
import do_mpc

class RotaryKilnMPC:
    def __init__(self):
        # 1. Model Tanımlama
        self.model = self._setup_model()
        
        # 2. Kontrolcü Tanımlama
        self.mpc = self._setup_controller(self.model)
        
        # 3. Simülatör Tanımlama (Test amaçlı, gerçek sistem yerine geçer)
        self.simulator = self._setup_simulator(self.model)

    def _setup_model(self):
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # Durum Değişkenleri (States)
        temp = model.set_variable(var_type='_x', var_name='temp', shape=(1,1))

        # Kontrol Girişleri (Manipulated Variables - MV)
        fuel = model.set_variable(var_type='_u', var_name='fuel', shape=(1,1))

        # Zamanla Değişen Parametreler (Time-Varying Parameters - TVP)
        # RL ajanı bu setpoint değerini manipüle edecek
        temp_setpoint = model.set_variable(var_type='_tvp', var_name='temp_setpoint')
        material_feed = model.set_variable(var_type='_tvp', var_name='material_feed')

        # Fiziksel Parametreler
        thermal_mass = 5500.0  # kg * Cp
        heat_value = 30000.0   # kJ/kg
        loss_coeff = 0.05      # Isı kaybı katsayısı

        # Diferansiyel Denklem: dT/dt
        dT_dt = (fuel * heat_value - loss_coeff * (temp - 25) - (material_feed * 1.1 * (temp - 100))) / thermal_mass
        
        model.set_rhs('temp', dT_dt)
        model.setup()
        return model

    def _setup_controller(self, model):
        mpc = do_mpc.controller.MPC(model)

        setup_settings = {
            'n_horizon': 20,
            't_step': 1.0,
            'n_robust': 1,
            'store_full_solution': True,
        }
        mpc.set_settings(**setup_settings)

        # Hedef Fonksiyonu: Sıcaklığı setpoint'te tut ve yakıtı optimize et
        lterm = (model.x['temp'] - model.tvp['temp_setpoint'])**2
        mterm = (model.x['temp'] - model.tvp['temp_setpoint'])**2
        
        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(fuel=1e-2) # Yakıt değişimlerini yumuşat

        # Kısıtlamalar (Constraints)
        mpc.bounds['lower', '_u', 'fuel'] = 0.5   # Min yakıt
        mpc.bounds['upper', '_u', 'fuel'] = 12.0  # Max yakıt
        mpc.bounds['lower', '_x', 'temp'] = 500.0 # Güvenlik alt sınırı
        mpc.bounds['upper', '_x', 'temp'] = 1600.0 # Erime riski sınırı

        # Zamanla değişen parametre fonksiyonu (Setpoint ve Feed bilgisini besler)
        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_now):
            # Varsayılan değerler, simülasyon sırasında güncellenecek
            tvp_template['_tvp', :, 'temp_setpoint'] = 1400.0
            tvp_template['_tvp', :, 'material_feed'] = 5.0
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        return mpc

    def _setup_simulator(self, model):
        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_settings(t_step=1.0)
        
        tvp_template = simulator.get_tvp_template()
        def tvp_fun(t_now):
            tvp_template['_tvp', 'temp_setpoint'] = 1400.0
            tvp_template['_tvp', 'material_feed'] = 5.0
            return tvp_template

        simulator.set_tvp_fun(tvp_fun)
        simulator.setup()
        return simulator

    def step(self, current_temp, target_temp, feed_rate):
        """
        Dış dünyadan (veya RL'den) gelen verilerle MPC'yi bir adım tetikler.
        """
        # Mevcut durumu güncelle
        self.mpc.x0 = np.array([current_temp])
        
        # Gelecek parametreleri ayarla (RL'den gelen hedef)
        # Not: Gerçek uygulamada tvp_fun üzerinden dinamik beslenir.
        
        # Kontrol sinyalini hesapla
        u_next = self.mpc.make_step(np.array([current_temp]))
        return u_next[0][0] # Önerilen yakıt miktarı
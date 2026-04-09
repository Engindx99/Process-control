import numpy as np
from casadi import *
import do_mpc

class AdvancedRotaryKilnMPC:
    def __init__(self):
        self.model = self._setup_model()
        self.mpc = self._setup_controller(self.model)
        self.x0 = np.array([1200.0])
        self.mpc.x0 = self.x0
        self.mpc.set_initial_guess()

    def _setup_model(self):
        model = do_mpc.model.Model('continuous')
        temp = model.set_variable('_x', 'temp')
        fuel = model.set_variable('_u', 'fuel')
        fan  = model.set_variable('_u', 'fan')
        temp_sp = model.set_variable('_tvp', 'temp_setpoint')
        feed = model.set_variable('_tvp', 'material_feed')

        # --- PARAMETRELER (Digital Twin ile Eşitlendi) ---
        thermal_mass = 5000.0
        heat_value = 32000.0 # DT ile aynı
        cp = 1.1
        k_conv = 0.08  # DT'deki yeni değer
        k_rad = 5e-10  # DT'deki yeni değer
        ambient = 25.0

        # --- Verim ve Isı Dengesi ---
        eta = 0.55 + 0.45 * tanh(fan / 40)
        q_in = fuel * heat_value * eta
        q_loss = k_conv * (temp - ambient) + k_rad * (temp**4 - ambient**4)
        q_material = feed * cp * (temp - 100)
        q_fan = 0.08 * fan * (temp - ambient) # Fan soğutma etkisi artırıldı

        dT_dt = (q_in - q_loss - q_material - q_fan) / (thermal_mass * cp)

        model.set_rhs('temp', dT_dt)
        model.setup()
        return model

    def _setup_controller(self, model):
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.n_horizon = 40 # Biraz daha ileriye baksın
        mpc.settings.t_step = 1.0
        mpc.settings.store_full_solution = True

        temp = model.x['temp']
        temp_sp = model.tvp['temp_setpoint']

        # Hedefe odaklanma ağırlığını artırdık
        lterm = (temp - temp_sp)**2
        mterm = (temp - temp_sp)**2
        mpc.set_objective(mterm=mterm, lterm=lterm)

        # Giriş değişim cezaları (R-term)
        # Yakıtın çok oynamasını istemiyoruz ama fanı kullanmasına izin veriyoruz
        mpc.set_rterm(fuel=0.5, fan=0.05)

        # --- KRİTİK SINIRLAR (CONSTRAINTS) ---
        # Yakıt alt limitini 0.0 yapıyoruz ki fırın soğuyabilsin!
        mpc.bounds['lower', '_u', 'fuel'] = 0.0  # Eskiden 0.5'ti, kilit buradaydı.
        mpc.bounds['upper', '_u', 'fuel'] = 12.0
        
        mpc.bounds['lower', '_u', 'fan'] = 10
        mpc.bounds['upper', '_u', 'fan'] = 100

        mpc.bounds['lower', '_x', 'temp'] = 500
        mpc.bounds['upper', '_x', 'temp'] = 1650

        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_now):
            tvp_template['_tvp', :, 'temp_setpoint'] = 1400.0
            tvp_template['_tvp', :, 'material_feed'] = 5.0
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        return mpc

    def step(self, measured_temp):
        self.mpc.x0 = np.array([measured_temp])
        u = self.mpc.make_step(self.mpc.x0)
        return float(u[0]), float(u[1])
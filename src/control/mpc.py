import numpy as np
from casadi import *
import do_mpc

class AdvancedRotaryKilnMPC:
    def __init__(self):
        self.model = self._setup_model()
        self.mpc = self._setup_controller(self.model)
        self.simulator = self._setup_simulator(self.model)

        # initial state
        self.x0 = np.array([1200.0])
        self.mpc.x0 = self.x0
        self.simulator.x0 = self.x0
        self.mpc.set_initial_guess()

    # -----------------------------
    # MODEL
    # -----------------------------
    def _setup_model(self):
        model = do_mpc.model.Model('continuous')

        # --- STATE ---
        temp = model.set_variable('_x', 'temp')

        # --- INPUTS ---
        fuel = model.set_variable('_u', 'fuel')
        fan  = model.set_variable('_u', 'fan')

        # --- TVP ---
        temp_sp = model.set_variable('_tvp', 'temp_setpoint')
        feed = model.set_variable('_tvp', 'material_feed')

        # --- PARAMETERS ---
        thermal_mass = 5500.0
        heat_value = 30000.0

        k_conv = 0.04
        k_rad = 1e-10
        ambient = 25.0

        # --- combustion efficiency approx ---
        eta = 0.7 + 0.3 * tanh(fan / 50)

        # --- heat terms ---
        q_in = fuel * heat_value * eta
        q_loss = k_conv * (temp - ambient) + k_rad * (temp**4 - ambient**4)
        q_material = feed * 1.1 * (temp - 100)
        q_fan = 0.02 * fan * (temp - ambient)

        # --- dynamics ---
        dT_dt = (q_in - q_loss - q_material - q_fan) / thermal_mass

        model.set_rhs('temp', dT_dt)
        model.setup()

        return model

    # -----------------------------
    # MPC CONTROLLER
    # -----------------------------
    def _setup_controller(self, model):
        mpc = do_mpc.controller.MPC(model)

        setup_mpc = {
            'n_horizon': 20,
            't_step': 1.0,
            'store_full_solution': True,
        }
        mpc.set_settings(**setup_mpc)

        # --- objective ---
        temp = model.x['temp']
        temp_sp = model.tvp['temp_setpoint']

        lterm = (temp - temp_sp)**2
        mterm = (temp - temp_sp)**2

        mpc.set_objective(mterm=mterm, lterm=lterm)

        # --- input penalty ---
        mpc.set_rterm(fuel=1e-2, fan=1e-3)

        # --- constraints ---
        mpc.bounds['lower', '_u', 'fuel'] = 0.5
        mpc.bounds['upper', '_u', 'fuel'] = 12.0

        mpc.bounds['lower', '_u', 'fan'] = 10
        mpc.bounds['upper', '_u', 'fan'] = 100

        mpc.bounds['lower', '_x', 'temp'] = 500
        mpc.bounds['upper', '_x', 'temp'] = 1600

        # --- TVP function ---
        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            tvp_template['_tvp', :, 'temp_setpoint'] = 1400.0
            tvp_template['_tvp', :, 'material_feed'] = 5.0
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        mpc.setup()
        return mpc

    # -----------------------------
    # SIMULATOR
    # -----------------------------
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

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, current_temp):
        self.mpc.x0 = np.array([current_temp])

        u = self.mpc.make_step(self.mpc.x0)

        fuel = u[0][0]
        fan  = u[1][0]

        return fuel, fan

    # -----------------------------
    # TEST SIMULATION
    # -----------------------------
    def run(self, steps=50):
        temp = self.x0[0]

        for i in range(steps):
            fuel, fan = self.step(temp)

            y_next = self.simulator.make_step(np.array([fuel, fan]))
            temp = y_next[0]

            print(f"Step {i}: Temp={temp:.2f}, Fuel={fuel:.2f}, Fan={fan:.2f}")
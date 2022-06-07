# pylint: disable=C0103
"""
This module contains common control functions and classes.

"""
import numpy as np
from sklearn.utils import Bunch
from helpers import complex2abc, abc2complex
from model.interfaces import solve


def state_machine(mdl, ctrl):
    """
    State machine for running the simulation.

    This runs the digital controller and calls the solver for integrating the
    continuous-time system model.

    Parameters
    ----------
    mdl : object
        System model.
    ctrl : object
        Controller.

    """
    def sensorless_ctrl():
        while mdl.t0 <= mdl.t_stop:
            # Sample the phase currents and the DC-bus voltage
            i_s_abc_meas = mdl.motor.meas_currents()
            u_dc_meas = mdl.converter.meas_dc_voltage()
            # Get the speed reference
            w_m_ref = mdl.speed_ref(mdl.t0)
            # Run the digital controller
            d_abc_ref, T_s = ctrl(w_m_ref, i_s_abc_meas, u_dc_meas)
            # Model the computational delay
            d_abc = mdl.delay(d_abc_ref)
            # Simulate the continuous-time plant model over the sampling period
            solve(mdl, d_abc, [mdl.t0, mdl.t0+T_s])

    def sensored_ctrl():
        while mdl.t0 <= mdl.t_stop:
            # Sample the phase currents and the DC-bus voltage
            i_s_abc_meas = mdl.motor.meas_currents()
            u_dc_meas = mdl.converter.meas_dc_voltage()
            # Measure the rotor position (not used in the case of an IM)
            theta_m_meas = ctrl.p*mdl.mech.meas_position()
            # Measure the rotor speed
            w_m_meas = ctrl.p*mdl.mech.meas_speed()
            # Get the speed reference
            w_m_ref = mdl.speed_ref(mdl.t0)
            # Run the digital controller
            d_abc_ref, T_s = ctrl(w_m_ref, i_s_abc_meas, u_dc_meas, w_m_meas,
                                  theta_m_meas)
            # Model the computational delay
            d_abc = mdl.delay(d_abc_ref)
            # Simulate the continuous-time plant model over the sampling period
            solve(mdl, d_abc, [mdl.t0, mdl.t0 + T_s])

    # Run the state machine
    if ctrl.sensorless:
        sensorless_ctrl()
    else:
        sensored_ctrl()


# %%
def duty_ratios(u, u_dc):
    """
    Compute the duty ratios for three-phase PWM.

    This compuates the duty ratios for three-phase PWM using a symmetrical
    suboscillation method. This method is identical to the standard
    space-vector PWM.

    Parameters
    ----------
    u : complex
        Voltage space vector (typically reference value).
    u_dc : float
        DC-bus voltage (typically measured value).

    Returns
    -------
    d_abc : ndarray, shape (3,)
        Duty ratios (typically reference values).

    """
    # Phase voltages without the zero-sequence voltage
    u_abc = complex2abc(u)
    # Symmetrization by adding the zero-sequence voltage
    u_0 = .5*(np.amax(u_abc) + np.amin(u_abc))
    u_abc -= u_0
    # Preventing overmodulation by means of a minimum phase error method
    m = (2./u_dc)*np.amax(u_abc)
    if m > 1:
        u_abc = u_abc/m
    # Duty ratios
    d_abc = .5 + u_abc/u_dc
    return d_abc


# %%
class PWM:
    """
    Compute the duty ratio references and the realized voltage.

    This contains the compuation of the duty ratio references and the realized
    voltage. The compuation of the realized voltage takes the digital delay
    effects into account.

    """

    def __init__(self, pars):
        """
        Parameters
        ----------
        pars : data object
            Controller parameters.

        """
        self.T_s = pars.T_s
        self.realized_voltage = 0
        self._u_ref_lim_old = 0

    def __call__(self, u_ref, u_dc, theta, w):
        """
        Compute the duty ratio references and update the state.

        Parameters
        ----------
        u_ref : complex
            Voltage reference in synchronous coordinates.
        u_dc : float
            DC-bus voltage.
        theta : float
            Angle of synchronous coordinates.
        w : float
            Angular frequency of synchronous coordinates.

        Returns
        -------
        d_abc_ref : ndarray, shape (3,)
            Duty ratio references.

        """
        d_abc_ref, u_ref_lim = self.output(u_ref, u_dc, theta, w)
        self.update(u_ref_lim)
        return d_abc_ref

    def output(self, u_ref, u_dc, theta, w):
        """
        Compute the duty ratio references and the limited voltage reference.

        """
        # Advance the angle due to the computational delay (T_s) and
        # the ZOH (PWM) delay (0.5*T_s)
        theta_comp = theta + 1.5*self.T_s*w
        # Voltage reference in stator coordinates
        u_s_ref = np.exp(1j*theta_comp)*u_ref
        # Duty ratios
        d_abc_ref = duty_ratios(u_s_ref, u_dc)
        # Realizable voltage
        u_s_ref_lim = abc2complex(d_abc_ref)*u_dc
        u_ref_lim = np.exp(-1j*theta_comp)*u_s_ref_lim
        return d_abc_ref, u_ref_lim

    def update(self, u_ref_lim):
        """
        Update the voltage estimate for the next sampling instant.

        Parameters
        ----------
        u_ref_lim : complex
            Limited voltage reference in synchronous coordinates.

        """
        self.realized_voltage = .5*(self._u_ref_lim_old + u_ref_lim)
        self._u_ref_lim_old = u_ref_lim


# %%
class SpeedCtrl:
    """
    2DOF PI speed controller.

    This speed controller is implemented using the disturbance-observer
    structure. The controller is mathematically identical to the 2DOF PI speed
    controller.

    """

    def __init__(self, pars):
        """
        Parameters
        ----------
        pars : data object
            Controller parameters.

        """
        self.T_s = pars.T_s
        self.alpha_s = pars.alpha_s
        self.tau_M_max = pars.tau_M_max
        self.J = pars.J
        self.k = pars.alpha_s*pars.J    # Gain
        self.tau_l = 0                  # Integral state
        self.desc = (('2DOF PI speed control:\n'
                      '    alpha_s=2*pi*{:.1f}\n')
                     .format(pars.alpha_s/(2*np.pi)))

    def output(self, w_M_ref, w_M):
        """
        Compute the torque reference and the load torque estimate.

        Parameters
        ----------
        w_M_ref : float
            Rotor speed reference (in mechanical rad/s).
        w_M : float
            Rotor speed (in mechanical rad/s).

        Returns
        -------
        tau_M_ref : float
            Torque reference.
        tau_L : float
            Load torque estimate.

        """
        tau_L = self.tau_l - self.alpha_s*self.J*w_M
        tau_M_ref = self.k*(w_M_ref - w_M) + tau_L
        if np.abs(tau_M_ref) > self.tau_M_max:
            tau_M_ref = np.sign(tau_M_ref)*self.tau_M_max
        return tau_M_ref, tau_L

    def update(self, tau_M, tau_L):
        """
        Update the integral state.

        Parameters
        ----------
        tau_M : float
            Realized (limited) torque reference.
        tau_L : float
            Load torque estimate.

        """
        self.tau_l += self.T_s*self.alpha_s*(tau_M - tau_L)

    def __str__(self):
        return self.desc


# %%
class RateLimiter:
    """
    Rate limiter.

    """

    def __init__(self, pars):
        """
        Parameters
        ----------
        pars : data object
            Controller parameters.

        """
        self.T_s = pars.T_s
        self.limit = pars.rate_limit
        self.y_old = 0

    def __call__(self, u):
        """
        Parameters
        ----------
        u : float
            Input signal.

        Returns
        -------
        y : float
            Rate-limited output signal.

        Notes
        -----
        In this implementation, the falling rate limit equals the (negative)
        rising rate limit. If needed, these limits can be separated with minor
        modifications in the class.

        """
        rate = (u - self.y_old)/self.T_s
        if rate > self.limit:
            # Limit rising rate
            y = self.y_old + self.T_s*self.limit
        elif rate < -self.limit:
            # Limit falling rate
            y = self.y_old - self.T_s*self.limit
        else:
            y = u
        # Store the limited output
        self.y_old = y
        return y


# %%
class Datalogger:
    """
    Datalogger for the control system.

    """

    def __init__(self):
        self.data = Bunch()

    def save(self, data):
        """
        Save the solution.

        Parameters
        ----------
        data : dictionary or Bunch object
            Data to be saved.

        """
        try:
            for key, value in data.items():
                self.data[key].extend([value])
        except KeyError:
            # Lists do not exist, initialize them
            for key, value in data.items():
                self.data[key] = []

    def post_process(self):
        """
        Transform the lists to the ndarray format.

        """
        for key in self.data:
            self.data[key] = np.asarray(self.data[key])
        try:
            # Compute time vector using the stored sampling periods
            self.data["t"] = np.cumsum(self.data["T_s"]) - self.data["T_s"][0]
        except KeyError:
            pass

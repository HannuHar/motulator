# pylint: disable=invalid-name
"""Continuous-time models for mechanical subsystems."""
import numpy as np

# %%
class Mechanics:
    """
    Mechanics subsystem.

    This models an equation of motion for stiff mechanics.

    Parameters
    ----------
    J : float
        Total moment of inertia.
    B : float
        Viscous damping coefficient.
    tau_L_ext : function
        External load torque as a function of time, `tau_L_ext(t)`.

    """

    def __init__(self, J=.015, B=0, mag_vib=0, f_vib=0, t_vib=np.Infinity, tau_L_ext=lambda t: 0):
        self.J, self.B = J, B
        self.tau_L_ext = tau_L_ext
        self.mag_vib, self.f_vib, self.t_vib = mag_vib, f_vib, t_vib
        # Initial values
        self.w_M0, self.theta_M0 = 0, 0

    def f(self, t, w_M, tau_M):
        """
        Compute the state derivative.

        Parameters
        ----------
        t : float
            Time.
        w_M : float
            Rotor angular speed (in mechanical rad/s).
        tau_M : float
            Electromagnetic torque.

        Returns
        -------
        list, length 2
            Time derivative of the state vector.

        """
        if t > self.t_vib:
            dw_M = (tau_M - self.tau_L_ext(t) + 
                    self.mag_vib*2*np.pi* self.f_vib*np.cos(2*np.pi* self.f_vib*t))/self.J
            dtheta_M = w_M + self.mag_vib*np.sin(2*np.pi* self.f_vib*t)
        else:
            dw_M = (tau_M - self.B*w_M - self.tau_L_ext(t))/self.J
            dtheta_M = w_M
        return [dw_M, dtheta_M]

    def meas_speed(self):
        """
        Measure the rotor speed.

        This returns the rotor speed at the end of the sampling period.

        Returns
        -------
        w_M0 : float
            Rotor angular speed (in mechanical rad/s).

        """
        # The quantization noise of an incremental encoder could be modeled
        # by means of the rotor angle, to be done later.
        return self.w_M0

    def meas_position(self):
        """
        Measure the rotor angle.

        This returns the rotor angle at the end of the sampling period.

        Returns
        -------
        theta_M0 : float
            Rotor angle (in mechanical rad).

        """
        return self.theta_M0

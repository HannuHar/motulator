# pylint: disable=invalid-name
"""Helper functions and classes."""

# %%
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt


# %%
def abc2complex(u):
    """
    Transform three-phase quantities to a complex space vector.

    Parameters
    ----------
    u : array_like, shape (3,)
        Phase quantities.

    Returns
    -------
    complex
        Complex space vector (peak-value scaling).

    Examples
    --------
    >>> from motulator import abc2complex
    >>> y = abc2complex([1, 2, 3])
    >>> y
    (-1-0.5773502691896258j)

    """
    return (2/3)*u[0] - (u[1] + u[2])/3 + 1j*(u[1] - u[2])/np.sqrt(3)


# %%
def complex2abc(u):
    """
    Transform a complex space vector to three-phase quantities.

    Parameters
    ----------
    u : complex
        Complex space vector (peak-value scaling).

    Returns
    -------
    ndarray, shape (3,)
        Phase quantities.

    Examples
    --------
    >>> from motulator import complex2abc
    >>> y = complex2abc(1-.5j)
    >>> y
    array([ 1.       , -0.9330127, -0.0669873])

    """
    return np.array([
        u.real, .5*(-u.real + np.sqrt(3)*u.imag),
        .5*(-u.real - np.sqrt(3)*u.imag)
    ])


# %%
@dataclass
class BaseValues:
    """
    Base values.

    Base values are computed from the nominal values and the number of pole
    pairs. They can be used, e.g., for scaling the plotted waveforms.

    """

    # pylint: disable=too-many-instance-attributes
    U_nom: float
    I_nom: float
    f_nom: float
    P_nom: float
    tau_nom: float
    p: int

    def __post_init__(self):
        """Compute the base values."""
        self.u = np.sqrt(2/3)*self.U_nom
        self.i = np.sqrt(2)*self.I_nom
        self.w = 2*np.pi*self.f_nom
        self.psi = self.u/self.w
        self.P = 1.5*self.u*self.i
        self.Z = self.u/self.i
        self.L = self.Z/self.w
        self.tau = self.p*self.P/self.w


# %%
class Sequence:
    """
    Sequence generator.

    The time array must be increasing. The output values are interpolated
    between the data points.

    Parameters
    ----------
    times : ndarray
        Time values.
    values : ndarray
        Output values.
    periodic : bool, optional
        Enables periodicity. The default is False.

    """

    # pylint: disable=too-few-public-methods
    def __init__(self, times, values, periodic=False):
        self.times = times
        self.values = values
        if periodic is True:
            self._period = times[-1] - times[0]
        else:
            self._period = None

    def __call__(self, t):
        """
        Interpolate the output.

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        float or complex
            Interpolated output.

        """
        return np.interp(t, self.times, self.values, period=self._period)


# %%
class Step:
    """Step function."""

    # pylint: disable=too-few-public-methods
    def __init__(self, step_time, step_value, initial_value=0):
        self.step_time = step_time
        self.step_value = step_value
        self.initial_value = initial_value

    def __call__(self, t):
        """
        Step function.

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        float
            Step output.

        """
        return self.initial_value + (t >= self.step_time)*self.step_value


# %%
# The Bunch class is from sklearn.utils:
#
#    https://github.com/scikit-learn/scikit-learn
#
# Its license is provided below.
#
# ----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2007-2021 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------
class Bunch(dict):
    """
    Container object exposing keys as attributes.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> from sklearn.utils import Bunch
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            # pylint: disable=raise-missing-from
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass

def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx

def fft_from_data(t: np.ndarray, data: np.ndarray):

    yf = fft.fftshift(fft.fft(data))*(t[1]-t[0])
    xf = fft.fftshift(fft.fftfreq(t.size, t[1]-t[0]))
    
    return xf, yf

def torq_W_diff(t: np.ndarray, tau_M: np.ndarray, w_M: np.ndarray, start: float, f_base: float):
    # Select data points
    time = np.delete(np.delete(t, np.where(start + 5/f_base <= t)),
                    np.where(t <= start))
    torq = np.delete(np.delete(tau_M, np.where(start + 5/f_base <= t)), 
                    np.where(t <= start))
    w =   np.delete(np.delete(w_M, np.where(start + 5/f_base <= t)), 
                    np.where(t <= start))
    # FFT
    xf, tau_yf = fft_from_data(time, torq)
    _, w_yf = fft_from_data(time, w) 
    
    # Find frequency of interest
    idx = find_nearest(xf,f_base)
    
    
    return xf[idx], tau_yf[idx], w_yf[idx]

def find_next_zero(start,f):
    
    def func(t,f):
        return np.cos(2*np.pi*f*t)
    loop_on = True
    t1 = start
    delta = (1/f)/5000

    while loop_on:
        t1=t1+delta
        if (func(t1,f)*func(t1+delta,f)<0):
            t1 = t1+delta/2
            loop_on=False
            return t1 + np.arccos(0)/(2*np.pi*f)
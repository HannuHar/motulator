# motulator
Open-Source Simulator for Motor Drives and Power Converters

Introduction
------------
This software includes simulation models for an induction motor, a synchronous
reluctance motor, and a permanent-magnet synchronous motor. Furthermore,
some simple control algorithms are included as examples. The motor models
are simulated in the continuous-time domain while the control algorithms run
in discrete time. The default solver is the explicit Runge-Kutta method of
order 5(4) from scipy.integrate.solve_ivp.

More detailed configuration can be done by editing the config files. There
are separate config files for a drive system and for its controller. For
example, pulse-width modulation (PWM) can be enabled in the drive system
config file. The example control algorithms aim to be simple yet feasible.
They have not been optimized at all.

Notes
-----
This is the very first version. No detailed testing has been carried out.
There can be bugs and misleading comments. Many interfaces will change in the
later versions.
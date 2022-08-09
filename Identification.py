from pyexpat import model
# motulator imports
import motulator.simulation as simulation
import motulator.control.im_vhz as vhz_control
import motulator.model.converter as converter
import motulator.model. im_drive as im_drive
import motulator.model.im as im
import motulator.model.ideal_mech as mechanics
import motulator.plots as plt
import motulator.helpers as helpers
#other imports
from dataclasses import dataclass
from time import time


@dataclass
class model_params:
    # Base values
    U_nom: float = 400
    I_nom: float = 81
    f_nom: float = 50
    P_nom: float = 45e3
    tau_nom: float = 291

    # Motor pars
    j: float = .49
    b: float = 0.0

    R_s: float = 0.06
    R_R: float = 0.03
    L_sig: float = 0.0022
    L_M: float = 0.0245
    p: int = 2

    l_unsat: float = .34
    beta: float = .84
    s: int = 7
    u_cons: float = 540
    def __post_init__(self):
        self.base = helpers.BaseValues(self.U_nom, self.I_nom, self.f_nom,
                                        self.P_nom, self.tau_nom, self.p)
        self.mech = mechanics.Mechanics(self.j, self.b)
        self.motor = im.InductionMotorInvGamma(self.p, self.R_s, self.R_R, self.L_sig, self.L_M)
        self.conv = converter.Inverter(self.u_cons)
        self.mdl = im_drive.InductionMotorDrive(self.motor, self.mech, self.conv)

def main():
    params = model_params()
    mdl = params.mdl
    ctrl = vhz_control.InductionMotorVHzCtrl(
    vhz_control.InductionMotorVHzCtrlPars(R_s=0, R_R=0, k_u=0, k_w=0))
    base = params.base

    ctrl.w_m_ref = lambda t: (t > .2)*(1.*base.w)
    mdl.mech.tau_L_ext = lambda t: (t > 1.)*base.tau_nom

    sim = simulation.Simulation(mdl, ctrl, 1, False)
    t_start = time()
    sim.simulate(t_stop=2)
    print('\nExecution time: {:.2f} s'.format((time() - t_start)))
        
    plt.plot(sim)


if __name__ == "__main__":
    main()


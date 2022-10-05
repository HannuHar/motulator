# simualtion tools
import motulator.simulation as simulation
import motulator.helpers as helpers
import motulator.plots as plot
# control methodes
import motulator.control.im_vhz as vhz_control
import motulator.control.obs_vhz as obs_control

# aimulation models
import motulator.model.converter as converter
import motulator.model. im_drive as im_drive
import motulator.model.im as im
import motulator.model.ideal_mech as ideal_mech
import motulator.model.mech as mech
#other imports
from dataclasses import dataclass
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import gc
import scipy.io as io


@dataclass
class model_pars:
    ideal_mech: bool = True
    # Base values
    U_nom: float = 400
    I_nom: float = 81
    f_nom: float = 50
    P_nom: float = 45e3
    tau_nom: float = 291

    # Motor pars
    j: float = .49
    b: float = 0.0

    R_s: float = 60e-3
    R_R: float = 30e-3
    L_sig: float = 2.2e-3
    L_M: float = 24.5e-3
    p: int = 2

    l_unsat: float = .34
    beta: float = .84
    s: int = 7
    u_cons: float = 540

    # signal injected
    mag_vib: float = 1
    f_vib: float = 50
    t_vib: float = 3.5


    def __post_init__(self):
        self.base = helpers.BaseValues(self.U_nom, self.I_nom, self.f_nom,
                                        self.P_nom, self.tau_nom, self.p)
        if self.ideal_mech:
            self.mech = ideal_mech.Mechanics(self.j, self.b, self.mag_vib, self.f_vib, self.t_vib)
        else:
            self.mech = mech.Mechanics(self.j, self.b)

        self.motor = im.InductionMotorInvGamma(self.p, self.R_s, self.R_R, self.L_sig, self.L_M)
        self.conv = converter.Inverter(self.u_cons)
        self.mdl = im_drive.InductionMotorDrive(self.motor, self.mech, self.conv)

@dataclass
class model_pars_2_2:
    ideal_mech: bool =  True
    # Base values
    U_nom: float = 400
    I_nom: float = 5
    f_nom: float = 50
    P_nom: float = 2.2e3
    tau_nom: float = 14.6

    # Motor pars
    j: float = .0155
    b: float = 0.0

    R_s: float = 3.7
    R_R: float = 2.1
    L_sig: float = 21e-3
    L_M: float = 224e-3
    p: int = 2

    l_unsat: float = .34
    beta: float = .84
    s: int = 7
    u_cons: float = 540

    # signal injected
    mag_vib: float = 1
    f_vib: float = 100
    t_vib: float = 3.5


    def __post_init__(self):
        self.base = helpers.BaseValues(self.U_nom, self.I_nom, self.f_nom,
                                        self.P_nom, self.tau_nom, self.p)
        if self.ideal_mech:
            self.mech = ideal_mech.Mechanics(self.j, self.b, self.mag_vib, self.f_vib, self.t_vib)
        else:
            self.mech = mech.Mechanics(self.j, self.b)

        self.motor = im.InductionMotorInvGamma(self.p, self.R_s, self.R_R, self.L_sig, self.L_M)
        self.conv = converter.Inverter(self.u_cons)
        self.mdl = im_drive.InductionMotorDrive(self.motor, self.mech, self.conv)


def obs_vhz(mag_vib=0.01,f_vib=1,t_vib=np.infty):
    # https://doi.org/10.1109/JESTPE.2021.3060583
    # Model
    pars = model_pars_2_2(mag_vib=mag_vib, f_vib=f_vib, t_vib=t_vib)
    mdl = pars.mdl
    base = pars.base
    
    # Control
    ctrl = obs_control.ObserverBasedVHz(
                obs_control.IMObsVHzPars(T_s=250e-6,i_s_max=base.i*3,
                                         psi_s_ref=base.psi,
                                         alpha_c=2*np.pi*20,
                                         alpha_o=2*np.pi*40,
                                         zeta_inf=0.7,
                                         alpha_f=2*np.pi*1,
                                         k_w=3,
                                         R_s=pars.R_s,
                                         R_R=pars.R_R,
                                         L_sgm=pars.L_sig,
                                         L_M=pars.L_M))
    
    # Ramp and speed
    ctrl.w_m_ref = lambda t: (t > .2)*(.8*base.w)
    mdl.mech.tau_L_ext = lambda t: (t > 1.5)*base.tau_nom*0.8
    
    # Simulation model
    sim = simulation.Simulation(mdl, ctrl, 1, False)
    
    # Actual simulation
    sim.simulate(t_stop=t_vib + 20/f_vib)
    
    return sim

def closed_loop(mag_vib=0.001,f_vib=1,t_vib=np.infty):
    # https://doi.org/10.1109/JESTPE.2021.3060583
    # Model
    pars = model_pars_2_2(mag_vib=mag_vib, f_vib=f_vib, t_vib=t_vib)
    mdl = pars.mdl
    base = pars.base
    
    # Control
    ctrl = vhz_control.InductionMotorVHzCtrl(
    vhz_control.InductionMotorVHzCtrlPars(  T_s=250e-6,
                                            R_s=pars.R_s,
                                            R_R=pars.R_R,
                                            L_M=pars.L_M,
                                            L_sgm=pars.L_sig,
                                            psi_s_nom=base.psi,
                                            k_u=1,
                                            k_w=3))
    
    # Ramp and speed
    ctrl.w_m_ref = lambda t: (t > .2)*(.8*base.w)
    mdl.mech.tau_L_ext = lambda t: (t > 1.5)*base.tau_nom*0.8
    
    # Simulation model
    sim = simulation.Simulation(mdl, ctrl, 1, False)
    
    # Actual simulation
    sim.simulate(t_stop=t_vib + 20/f_vib)
    
    return sim     

def open_loop(mag_vib=0.001,f_vib=0.05,t_vib=np.infty, plot_data=False):
    # https://doi.org/10.1109/JESTPE.2021.3060583
    # Model
    pars = model_pars_2_2(mag_vib=mag_vib, f_vib=f_vib, t_vib=t_vib,ideal_mech=True)

    mdl = pars.mdl
    base = pars.base
    
    # Control
    ctrl = vhz_control.InductionMotorVHzCtrl(
    vhz_control.InductionMotorVHzCtrlPars(  
                                            R_s=pars.R_s,
                                            R_R=pars.R_R,
                                            L_M=pars.L_M,
                                            L_sgm=pars.L_sig,
                                            psi_s_nom=base.psi,
                                            k_u=0,
                                            k_w=0))
    
    # Ramp and speed
    ctrl.w_m_ref = lambda t: (t > .2)*(.8*base.w)
    mdl.mech.tau_L_ext = lambda t: (t > 1.5)*base.tau_nom*0.8
    
    # Simulation model
    sim = simulation.Simulation(mdl=mdl, ctrl = ctrl, delay=1, pwm=False)
    
    # Actual simulation
    sim.simulate(t_stop=t_vib + 20/f_vib)
    
    return sim   
    

def ident(i,f_vib,control,plot_data=False):
    t_vib = helpers.find_next_zero(12,f_vib)

    if control == "closed_loop":
        sim = closed_loop(f_vib=f_vib, t_vib=t_vib)
    if control == "open_loop":
        sim = open_loop(f_vib=f_vib, t_vib=t_vib)
    if control == "obs":
        sim = obs_vhz(f_vib=f_vib, t_vib=t_vib)
    
    # FFT
    f, tau_yf, w_yf =helpers.torq_W_diff(sim.mdl.data.t,
                                            sim.mdl.data.tau_M,
                                            sim.mdl.data.w_M,
                                            t_vib+13/f_vib,
                                            f_vib)
    if plot_data:
        print(f,tau_yf,w_yf)
        plot.plot(sim)
    
    # Clean up for memomry
    del sim
    gc.collect()
    
    return [i,f,tau_yf,w_yf]

def multi(control = "obs"):
    res = list()
    # Target folder:
    folder = "mat_files/" + "2_2kW/" +control + "/"
    # Frequency range
    freq = np.logspace(1,2,16)

    # data collection function for async multiprocessing
    def collect_result(result):
        res.append(result)
    

    # Multiprocessing
    pool = mp.Pool(mp.cpu_count())
    
    for i, x in enumerate(freq):
        pool.apply_async(ident, args=[i,x,control,False], callback=collect_result) 
    
    pool.close()
    pool.join()

    # sort data
    res.sort(key=lambda x: x[0])
    
    output = np.vstack(res)
    f = np.real(output[:,1])
    tau_M = output[:,2]
    w_M = output[:,3]

    # Save data
    mdic = {"f":f,"tau_M":tau_M,"w_M":w_M}
    io.savemat(folder + "pwm_"+
                str(freq.size) + "_samples_" +                      # name contains sample count and frequency range
                str('%.2E' % freq[0]).replace(".","_") +            # min freq
                "_to_" + 
                str('%.2E' % freq[-1]).replace(".","_") + "_Hz" +   # max freq
                ".mat", mdic)                                       # data format
    
    del res
    del output
    gc.collect()

    
if __name__ == "__main__":
    t_start = time()
    # multi("closed_loop")
    # multi("obs")
    multi("open_loop")
    # ident(1,50,"closed_loop",True)
    print('\nExecution time: {:.2f} s'.format((time() - t_start)))

    

       

    


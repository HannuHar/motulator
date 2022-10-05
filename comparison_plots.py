import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np

my_dpi = 120

folder = "mat_files/2_2kW/open_loop/"
mat = io.loadmat(folder + "pwm_16_samples_1_00E+01_to_1_00E+02_Hz" + ".mat")
f = mat["f"].T
tau_M = mat["tau_M"].T
w_M = mat["w_M"].T
tau_M.shape
resp = tau_M/(w_M*2)

w = io.loadmat(folder + "FRF_w.mat")["w"].T
H = io.loadmat(folder + "FRF_H.mat")["H"].T

data_amp = dict(xlabel=r"f [Hz]", ylabel=r"Amplitude [Nm/(Rad/s)]")
data_ang = dict(xlabel=r"f [Hz]", ylabel=r"Phase Shift [Deg]")
legend_pars = dict(frameon=True,facecolor="white",edgecolor="black",framealpha=1)

with plt.style.context(['science', 'ieee']):

    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(800/my_dpi, 650/my_dpi), dpi=my_dpi)

    
    ax1.plot(w/(2*np.pi),np.abs(H),label="Analytical")
    ax1.plot(f,np.abs(resp),label="Identified")
    ax1.loglog()
    ax1.legend(frameon=True,facecolor="white",edgecolor="black",framealpha=1)
    ax1.set(**data_amp)
    ax1.set_xlim([f[0],f[-1]])
    ax1.grid(True, which="both")

    ax2.plot(w/(2*np.pi),np.angle(H, deg=True),label="Analytical")
    ax2.plot(f, np.angle(-resp, deg=True),label="Identified")
    ax2.semilogx()
    ax2.legend(frameon=True,facecolor="white",edgecolor="black",framealpha=1)
    ax2.set(**data_ang)
    ax2.set_xlim([f[0],f[-1]])
    # ax2.set_ylim([-100,0])
    ax2.grid(True, which="both", axis="x")
    ax2.grid(True, which="major", axis="y")



plt.savefig("figures/closed_loop_2_2kW.pdf",)
plt.show()

# def model(x, p):
#     return x ** (2 * p + 1) / (1 + x ** (2 * p))


# pparam = dict(xlabel='Voltage (mV)', ylabel='Current ($\mu$A)')

# x = np.linspace(0.75, 1.25, 201)


# with plt.style.context(['science', 'ieee']):
#     fig, ax = plt.subplots()
#     for p in [10, 20, 40, 100]:
#         ax.plot(x, model(x, p), label=p)
#     ax.legend(**legend_pars)
    # legend.draw_frame(2000)
    # ax.autoscale(tight=True)
    # ax.grid(True)
    # ax.set(**pparam)
    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current (\textmu A)')  
    # fig.savefig('figures/fig2a.pdf')
    # fig.savefig('figures/fig2a.jpg', dpi=300)
# plt.show()
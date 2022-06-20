import numpy as np
from scipy.integrate import odeint
from scipy import interpolate
import matplotlib.pyplot as plt
neqs = 14
Vtemp = 13. # kT/Ze : temperature equivalent voltage for Ca ions (2+) - mV */

θ=[-61.93402836804239, 0.053999984513550134, 0.8677675132377433, 0.21094758823620002, 0.8673893800823501, 0.06463106009370051, 0.5368278680702148, 0.034801309766394395, 0.2629320245028402, 0.18635713309843333, 0.992863241496597, 0.9879736378591375, 0.5639252977789222, 0.7703557770831406, 0.3999940134854983, 150.65112731743588, 8.925651590974955, 49.086890011293676, 124.46183364682842, 0.7326262126952293, 1.5787349674992859, 3.7808472023523008, -66.22008947137167, 0.012024239193339614, -105.61790029997763, 0.6471293077332771, 8.08863953487089, 0.20724216148907304, 0.2728691459839464, 14.531212521513075, -0.054064741667474436, -22.78305852040714, -0.7153235615665068, -15.533841412033809, -86.08017380269727, -8.534653707506266, -0.11176514742260224, -43.4579767881865, 28.690779089823195, -0.00012204864425768847, -99.99664591194461, 31.775823913365354, -24.78067041995729, 6.005340205277638, 8.63823861520788, -10.087948987067847, 10.862059490607107, 33.81870582735651, -22.094680223135065, 10.147666479022288, 55.10854255397067, -28.051128176925005, 8.291500889922348, 25.575861938775848, 13.759408712422493, 11.388013273642223, -46.93101265686462, 16.32587595005947, 93.31552392777911, 37.665821487984786, 29.67668073324702, 9.580115826685166, 20.18556370676597, 0.02354327054488278, 0.19755436289318964, 0.14056010764790958, 0.6789422014322583, 176.39691337734806, 0.020257220783194835, 1.2471745968859276, 53.28620013766909, 0.16081583040084538, 0.030454186084830095, 1.8974888113214765, 0.06494438535366333, 0.5733558157973574, 0.5682582149312141, 0.1481098706558182, 103.43239161540912, 7.265638983689308, 4.057076960728887, 79.23928255912374, 33.06275563454173, 42.99153018700545, 197.689392415391, 914.1897391966728, 3.2050352920928056, 46.321464493541086]

path_to_stim_file = "making_stimulus_protocols/range=100pA/I_colpitts_x_time_dilation=1.0range=100_(I).txt"
ys = θ[0:14]
tstart = 1200. * 0.02
tend = 1400
num_tsteps = 1300
t_arr_solve = np.linspace(tstart, tend, num_tsteps)

stimulus = np.loadtxt(path_to_stim_file)
stimulus /= np.max(stimulus)
stimulus *= 1000.0

indices_stimulus = stimulus.shape[0]
dt = 0.02 # milliseconds
t_stim_arr = np.linspace(0,indices_stimulus*dt, indices_stimulus)
print(indices_stimulus)
print(t_stim_arr)
print(stimulus.shape)
print(t_arr_solve)
Iapp = interpolate.interp1d(t_stim_arr, stimulus)

def albeta(VV, V, dV, dVt, aV4, t0, τϵ, delta):
    # calculation of alpha and beta variable given the membrane voltage VV */
    #    int j;
    #    double thetai,thetait,tauj,thetai2,thetai1;
    alpha = np.zeros(14)
    beta = np.zeros(14)
    for j in range(1,neqs):
        thetai = (VV-V[j])/dV[j]
        thetait = (VV-V[j])/dVt[j]
        if j==5 or j==7 : # A2 and K2 tau_h */
            tauj = t0[j]+delta[j]+0.5*(1-np.tanh(1000*(VV-V[j]-aV4[j])))*(τϵ[j]*(1-np.tanh(thetait)*np.tanh(thetait))-delta[j]);
        else:
            if (j == 10): #T tau_h */
                thetai2 = (VV-V[j])/dVt[j]
                thetai1 = (VV-V[j])/aV4[j]
                tauj = t0[j]+τϵ[j]*(1+np.tanh(thetai2))*(1-np.tanh(thetai1))*(1-np.tanh(1000*(VV-V[j]))*np.tanh(thetai2+thetai1))/(1+np.tanh(thetai2)**2)
            else:
                tauj = t0[j]+τϵ[j]*(1 - np.tanh(thetait)*np.tanh(thetait));
        alpha[j] = 0.5*(1+np.tanh(thetai))/tauj
        beta[j] = 0.5*(1-np.tanh(thetai))/tauj
    return alpha, beta

def Nogaret_N2(yy, t, p):
    Ccap, gNa, gNaP, ENa, gA1, gA2, gK2, gC, EK, gL, EL, gou, gin, gh, Iarea, rlt = p[0:16]

    V1, V7, V8 = 0.0, -43., -59.
    V2, V3, V4, V5, V6, V9, V10, V11, V12, V13, V14 = p[16:27]
    V = [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14]

    dV1, dV7, dV8 = 0.0, +34., -21.
    dV2, dV3, dV4, dV5, dV6, dV9, dV10, dV11, dV12, dV13, dV14 =p[27:38]
    dV = [dV1, dV2, dV3, dV4, dV5, dV6, dV7, dV8, dV9, dV10, dV11, dV12, dV13, dV14]

    dVt1, dVt7, dVt8 = 0.0, +32., +25.
    dVt2, dVt3, dVt4, dVt5, dVt6, dVt9, dVt10, dVt11, dVt12, dVt13, dVt14 = p[38:49]
    dVt = [dVt1, dVt2, dVt3, dVt4, dVt5, dVt6, dVt7, dVt8, dVt9, dVt10, dVt11, dVt12, dVt13, dVt14]

    t01, t07, t08 = 0.0, 9.9, 50.
    t02, t03, t04, t05, t06, t09, t010, t011, t012, t013, t014 = p[49:60]
    t0 = [t01, t02, t03, t04, t05, t06, t07, t08, t09, t010, t011, t012, t013, t014]

    τϵ1, τϵ7, τϵ8 = 0.0, 66., 530.
    τϵ2, τϵ3, τϵ4, τϵ5, τϵ6, τϵ9, τϵ10, τϵ11, τϵ12, τϵ13, τϵ14 = p[60:71]
    τϵ = [τϵ1, τϵ2, τϵ3, τϵ4, τϵ5, τϵ6, τϵ7, τϵ8, τϵ9, τϵ10, τϵ11, τϵ12, τϵ13, τϵ14]

    delta = np.array([0, 0, 0, 0, 0, 0, 0, 450., 0, 0, 10., 0, 0, 0])
    aV4 = np.array([0, 0, 0, 0, 0, 0.790126, 0, 0, 0, 0, 5, 0, 0, 0])

    delta[5] = p[71]
    aV4[5], aV4[10] = p[72], p[73]

    alpha, beta = albeta(yy[0], V, dV, dVt, aV4, t0, τϵ, delta)

    yy1 = yy[0] / Vtemp
    yy2 = 1 + yy1 / 51  # Computes the (exp(x)-1)/x function to avoid divergence */
    for k in np.arange(49, 1, step=-1):
        yy2 = 1. + (float(yy1) / k) * yy2

    dyydx = np.zeros(neqs)
    dyydx[0] = (gNa * (yy[1] ** 3) * yy[2] + gNaP * yy[3]) * (ENa - yy[0])
    dyydx[0] += (gA1 * (yy[12] ** 4) + gA2 * (yy[4] ** 4) * yy[5] + gK2 * (yy[6] ** 4) * yy[7] + gC * yy[11]) * (EK - yy[0])
    dyydx[0] += (rlt * (yy[8] ** 2) + (yy[9] ** 2) * yy[10]) * Vtemp * (gou - gin * np.exp(yy1)) / yy2
    dyydx[0] += gL * (EL - yy[0]) + gh * yy[13] * (-43 - yy[0]) + (Iapp(t) - yy[0] / 1550) / Iarea  # was yy[0]/2260
    dyydx[0] /= Ccap
    for k in range(1, neqs):
        dyydx[k] = alpha[k] * (1 - yy[k]) - beta[k] * yy[k]
    return dyydx
print("Length of θ::" +str(len(θ)))
sol = odeint(Nogaret_N2, ys, t_arr_solve, args=(θ[neqs:],))

print(sol.shape)
print("DONE!")
plt.figure()
for channel in range(sol.shape[1]):
    plt.plot(sol[:,channel])
plt.ylim(-100,30)
plt.show()
plt.figure()
plt.plot(stimulus)
plt.show()

# path_to_obs = "datafiles/N2_volt.txt"
# nstart = Int(tstart/0.02)
# nend = Int(tend/0.02)
# data_t, data_u = load_data(path_to_obs, nstart, nend)
# plots = plot(data_t, data_u,
#              label = "Data", layout = (2, 1), legend=false, c="black")
# plot!(plots[2], sol.t, Iapp.(sol.t), c="blue", alpha=0.5)
# ylabel!(plots[1], "Voltage (mV)")
# ylabel!(plots[2], "Stimulus (nA)")
# xlabel!(plots[2], "Time (ms)")
# title!(plots[1], "HVCX Data")
# # savefig(plots, "X_proj_neuron_data.pdf")
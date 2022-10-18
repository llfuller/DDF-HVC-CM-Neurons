import numpy as np
import matplotlib.pyplot as plt

# Written October 11 by Lawson Fuller

loaded_file = np.load("conv_MSE_R_trials_array.npy")
print(loaded_file.shape) # dim(R, sigma, trial)

# num(R)=21
# num(sigma)=8
# num(trials)=30

sigma_0_arr = loaded_file[:,0,:]
sigma_5_arr = loaded_file[:,1,:]
sigma_1000_arr = loaded_file[:,5,:]
sigma_5000_arr = loaded_file[:,7,:]

np.savetxt("sigma_0_arr.txt",sigma_0_arr)
np.savetxt("sigma_5_arr.txt",sigma_5_arr)
np.savetxt("sigma_1000_arr.txt",sigma_1000_arr)
np.savetxt("sigma_5000_arr.txt",sigma_5000_arr)

np.savetxt('log10(R_values).txt',np.array([-6,-5.75,-5.5,-5.25,-5,-4.75,-4.5,-4.25,-4,-3.75,-3.5,-3.25,-3,-2.75,-2.5,-2.25,-2,-1.75,-1.5,-1.25,-1]))
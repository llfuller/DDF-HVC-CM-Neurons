import numpy as np

"""
Convert units in already-existing files to other units
"""

# Choose whether to convert to other units for output
convert_pA_to_nA = True
convert_s_to_ms = True
conversion_pA_to_nA = np.power(0.001,int(convert_s_to_ms))
conversion_s_to_ms = np.power(1000.0,int(convert_s_to_ms))

directory = "making_stimulus_protocols/range=700pA/"
filename = "I_L63_x_time_dilation=6.0range=700_(I)"

time_and_current = np.loadtxt(directory+filename+".txt")
t_arr = time_and_current[:,0]
I_arr = time_and_current[:,1]

t_arr = np.multiply(t_arr, conversion_s_to_ms)
I_arr = np.multiply(I_arr, conversion_pA_to_nA)

np.savetxt(directory+filename+"_converted_units.txt",
           np.concatenate((t_arr[:, np.newaxis], I_arr[:, np.newaxis]), axis=1))




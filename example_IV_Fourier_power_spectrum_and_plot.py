import numpy as np
from current_voltage_Fourier_power_spectrum_and_plot import produce_IV_Fourier_power_spectrum_and_plot

"""
Demonstration of how to correctly call 
current_voltage_Fourier_power_spectrum_and_plot.produce_IV_Fourier_power_spectrum_and_plot()
"""

example = 2 # choose which example to run

if example == 1:
    # Basic Plotting
    save_and_or_display = "display"
    neuron_number = "12"
    episode_number = "00023"
    a_filename_pretext = "nidaq_Dev1_ai-"+episode_number #everything in filename except "_voltage.png" for instance
    neuron_directory = "HVC_ra_x_i_data_2015/50 KhZ Recordings/Neuron "+neuron_number
    V = np.loadtxt(neuron_directory +"/"+ a_filename_pretext+"_voltage.txt")
    I = np.loadtxt(neuron_directory +"/"+ a_filename_pretext+"_current.txt")
    freq_units = "kHz" # frequency units

    sampling_rate = 50.0 # kHz (1/seconds)
    delta_t = 1.0/sampling_rate  # this is milliseconds (1.0/50kHz= 0.02ms)
    num_timesteps = V.shape[0]
    t_final = num_timesteps * delta_t  # measured in ms
    t_arr = np.arange(start=0, stop= t_final, step= delta_t) # units: ms

    # freq = 0.001 # units: 1/ms = kHz
    # V = np.sin(np.multiply(2*np.pi*(freq),t_arr))

    # Reduce array to only important timesteps to be analyzed. Will be removed from V and I and/or set to nan in later steps
    remove_indices = True
    range_list = [range(116000,t_arr.shape[0]),
                      range(20800,29700),
                      range(0,9100)
                  ]

    # Finally, produce the I(t) and V(t) plots, and their Fourier power spectrum plots:
    produce_IV_Fourier_power_spectrum_and_plot(V,I,range_list,t_arr,a_filename_pretext,freq_units,
                                               neuron_name=neuron_number, remove_indices=remove_indices, xlim=xlim,
                                               save_and_or_display = save_and_or_display)
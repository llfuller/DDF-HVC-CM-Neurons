import numpy as np
import save_utilities
import matplotlib.pyplot as plt
"""
Fourier power spectrum utility scripts.
Warning: Use the "compact" script. This script should be deprecated.
"""

def all_ndarrays_have_same_shape(ndarray_list):
    """Returns boolean reflecting whether ndarrays within ndarray_list have the same shape
    # Called by calc_Fourier_power_spectrum()
    # Input: ndarray_list
    """
    old_shape = None
    for an_array in ndarray_list:
        if old_shape != None:
            current_shape = an_array.shape
            if old_shape != current_shape:
                return False
        else: # if old_shape is None, AKA first iteration
            old_shape = an_array.shape
    return True

def calc_Fourier_power_spectrum(ndarray_list,t):
    """
    Calculates the Fourier power spectrum given two inputs ( [ndarray1,ndarray2,...], t )
    # Inputs: [ndarray_list, t]
    #   ndarray_list: list of numpy ndarrays (usually [current_array, voltage_array])
    #   t           : (numpy 1-D array) times of measurements
    # Returns: [FPS_list, frequency_array]
    #   [FPS_list containing FPS of multiple numpy arrays, frequency_range]
    #   Usually [ [FPS_current,FPS_voltage], frequency_range]
    #   Any one element from FPS_list with the array of frequency_range can be used to plot the Fourier power spectrum.
    """
    assert all_ndarrays_have_same_shape(ndarray_list) # Very important. Do not delete this line.
    sampling_rate = 1.0 / float(t[2] - t[1])
    FPS_list = []
    for an_array in ndarray_list:
        fourier_transform = np.fft.rfft(an_array)
        abs_fourier_transform = np.abs(fourier_transform)
        power_spectrum = np.square(abs_fourier_transform)
        frequency_array = np.linspace(0, sampling_rate / 2, len(power_spectrum))
        FPS_list.append(power_spectrum)
    return [FPS_list, frequency_array]

def plot_Fourier_power_spectrum(FPS_list, frequency_array, a_filename, directory_to_store_plots,
                                directory_to_store_txt_data, neuron_name, current_or_voltage, x_units,
                                xlim=None,ylim=None,save_and_or_display = "save", use_log = False):
    """
    Plots (and by default saves) the Fourier power spectrum for input list of FPS arrays
    # Inputs: FPS_list, frequency_array, a_filename, directory_to_store_plots,
    #                             directory_to_store_txt_data, neuron_name, save_and_or_display
    # Returns: Nothing, but can save and/or display data depending on save_and_or_display.
    """
    for i, power_spectrum in enumerate(FPS_list):
        freq_without_0_index = frequency_array[1:]
        normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if use_log == False:
            ax.plot(freq_without_0_index, normalized_power_spec_without_0_index)
            ax.set_ylabel("Normalized Power (1.0 = max from [1:])")
        if use_log == True:
            ax.plot(freq_without_0_index, np.log(normalized_power_spec_without_0_index))
            ax.set_ylabel("Log(Normalized Power (1.0 = max from [1:]))")

        print(a_filename[:-4])
        ax.set_title("Power(freq) of "+current_or_voltage[i]+"_train (Neuron " + str(neuron_name) + '_' + str(
            a_filename[:-4]) + ")")
        ax.set_xlabel("Frequency ("+str(x_units)+")")


        if xlim != None: ax.set_xlim(left=xlim[0],right=xlim[1])
        if ylim != None: ax.set_ylim(left=ylim[0],right=ylim[1])

        fig.show()

        save_utilities.save_and_or_display_plot(figure=fig,
                                                a_str=save_and_or_display,
                                                save_location=directory_to_store_plots + "Power spectrum of training "+
                                                              current_or_voltage[i]+" (Neuron " +
                                                              str(neuron_name + '_' + str(a_filename[:-4]) +
                                                                  ")_full_data.png"))
        save_utilities.save_text(data=np.column_stack((frequency_array, power_spectrum)),
                                      a_str = save_and_or_display.lower(),
                                      save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(
                                          a_filename[:-4]) + "_Fourier_Spectrum_training_"+current_or_voltage[i]+
                                                      "-full_data.txt")

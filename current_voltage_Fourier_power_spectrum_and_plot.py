import Fourier_Power_Spectrum as fps
import plotting_utilities
import array_utilities
import matplotlib.pyplot as plt
import numpy as np
import copy



def produce_IV_Fourier_power_spectrum_and_plot(V,I,range_list,t_arr,a_filename_pretext,freq_units,neuron_name,
                                               remove_indices=False, xlim=None, save_and_or_display="display",
                                               current_units = "Unspecified", voltage_units = "Unspecified"):
    """
        Call and loop over this function for different files.

        inputs: V, I, range_list, t_arr, a_filename_pretext, freq_units, neuron_name,
                remove_indices=False, save_and_or_display="display"
        returns: No returns

    """
    if freq_units == "kHz": time_units = "ms"
    if freq_units == "Hz": time_units = "s"
    V_reduced = copy.copy(V)
    I_reduced = copy.copy(I)
    if remove_indices == True:
        V_reduced = array_utilities.remove_elements(V, range_list)
        I_reduced = array_utilities.remove_elements(I, range_list)
        t_arr_reduced_nan_version = array_utilities.replace_ranges_with_nan(t_arr,range_list) # units: ms
    else:
        t_arr_reduced_nan_version = copy.copy(t_arr)

    # Plot Voltage and Current with x-axis indices from range_list missing
    plotting_utilities.plotting_quantity(x_arr=t_arr_reduced_nan_version, y_arr=I,
                                         title="Current ("+a_filename_pretext+")",xlabel="Time ("+str(time_units)+")",
                                         ylabel="Current ("+str(current_units)+")", save_and_or_display=save_and_or_display)
    plotting_utilities.plotting_quantity(x_arr=t_arr_reduced_nan_version, y_arr=V,
                                         title="Voltage ("+a_filename_pretext+")", xlabel="Time ("+str(time_units)+")",
                                         ylabel="Voltage ("+str(voltage_units)+")", save_and_or_display=save_and_or_display)

    # Perform Fourier power spectrum calculation
    [FPS_list, frequency_array] = fps.calc_Fourier_power_spectrum([I_reduced,V_reduced], t_arr)

    # Plot Fourier power spectrum
    fps.plot_Fourier_power_spectrum(FPS_list=FPS_list,
                                    frequency_array=frequency_array,
                                    a_filename=a_filename_pretext+".png",
                                    directory_to_store_plots=".",
                                    directory_to_store_txt_data=".",
                                    neuron_name=neuron_name,
                                    current_or_voltage = ["current","voltage"],
                                    x_units=freq_units,
                                    xlim = xlim, # kHz
                                    ylim = None,
                                    save_and_or_display="display")
    return [FPS_list, frequency_array]
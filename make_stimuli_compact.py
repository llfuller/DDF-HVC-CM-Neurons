import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the "making_stimulus_protocols" folder to the system path
making_stimulus_protocols_path = os.path.join(os.path.dirname(__file__), "making_stimulus_protocols")
sys.path.insert(1, making_stimulus_protocols_path)

# Import the externally_provided_currents module
import externally_provided_currents

# Add the "data_to_figure_code" folder to the system path
data_to_figure_code_path = os.path.join(os.path.dirname(__file__), "data_to_figure_code")
sys.path.insert(1, data_to_figure_code_path)

# Import other required functions
from make_rescaled_array import rescale_array
from make_FPS_compact import Fourier_Power_Spectrum_plot_and_save
from make_general_plot import time_series_plot_and_save


"""
Makes chaotic stimuli of various combinations of frequencies amplitudes.
Saves text and plots png of stimuli and saves text and png of FPS of stimuli.
In order to make the original data sent to Arij, the input lower_bound in rescale_array is always -100.
Example: I_L63_x_rescaled = rescale_array(I_L63_x, -100, upper_bound)
Version of make_stimuli.py cleaned up with GPT4 assistance.
"""

def process_currents(currents, name, suffixes, dilation_factor, lower_bound, upper_bound, t_arr, dt, save_directory):
    """
    Process and plot the time series and Fourier Power Spectrum of the input currents.

    Args:
        currents (list): List of input current arrays.
        name (str): Base name for the currents (e.g., "L63", "Colpitts", "L96").
        suffixes (list): List of suffixes for the currents (e.g., ["x", "y", "z"] or ["x1"]).
        dilation_factor (float): Time dilation factor used in scaling the currents.
        upper_bound (int): Upper bound used in rescaling the currents.
        t_arr (numpy.ndarray): Time array used for plotting the time series.
        dt (float): Time step for the Fourier Power Spectrum calculations.
    """
    rescaled_currents = [rescale_array(current, lower_bound, upper_bound) for current in currents]

    folder_path = save_directory + "(lower,upper)=" + str((float("{:.2f}".format(lower_bound)),upper_bound)) +"pA/"

    # Create the directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, rescaled_current in enumerate(rescaled_currents):
        suffix = suffixes[i]
        bounds_str = f"(lower,upper)=({float('{:.2f}'.format(lower_bound))},{upper_bound})pA"

        time_series_plot_and_save(rescaled_current, dt, folder_path,
                                  title=rf"${name}_{suffix}\ time\ dilation=" + str(dilation_factor) + "$",
                                  save_filename=f"I_{name}_{suffix}_time_dilation=" + str(
                                      dilation_factor) + f"_{bounds_str}",
                                  t_array=t_arr, xlabel="Time (seconds)", ylabel="Current (pA)")

        # save txt data of current
        np.savetxt(folder_path + f"I_{name}_{suffix}_time_dilation=" + str(dilation_factor) + f"_{bounds_str}.txt",
                   np.column_stack((t_arr, rescaled_current)))

        Fourier_Power_Spectrum_plot_and_save(rescaled_current, f"{name}_{suffix}_rescaled", sampling_rate=1.0 / dt,
                                             save_folder=folder_path,
                                             title=rf"$I\_{name}_{suffix}\ time\ dilation=" + str(
                                                 dilation_factor) + "$",
                                             save_filename=f"{name}_{suffix}_rescaled_FPS_time_dilation=" + str(
                                                 dilation_factor) + f"_{bounds_str}", xlim=60)

def main():

    t_initial = -1  # seconds
    t_final = 16  # seconds
    dt = 0.00002  # seconds
    t_arr = np.arange(t_initial, t_final, dt)
    save_directory = "stimuli_April_2023/"


    for dilation_factor in [5]:
        print("Dilation factor:"+str(dilation_factor))

        # for upper_bound in [100, 200, 300, 400, 500, 600, 700]:
        #     print("Upper bound: "+str(upper_bound))
        #     lower_bound = -float(upper_bound)/3 # = -100 for Arij data
        #
        #     # L63 time dilation and processing
        #     scaling_time_L63 = dilation_factor * 22.0
        #     L63_obj = externally_provided_currents.L63_object(scaling_time_factor=scaling_time_L63)
        #     I_L63 = L63_obj.prepare_f(t_arr).T
        #     process_currents(I_L63, "L63", ["x", "y", "z"], dilation_factor, lower_bound, upper_bound, t_arr, dt, save_directory)
        #
        #     # Colpitts time dilation and processing
        #     scaling_time_Colpitts = dilation_factor * 150.0
        #     colp_obj = externally_provided_currents.Colpitts_object(scaling_time_factor=scaling_time_Colpitts)
        #     I_colpitts = colp_obj.prepare_f(t_arr).T
        #     process_currents(I_colpitts, "Colpitts", ["x", "y", "z"], dilation_factor, lower_bound, upper_bound, t_arr, dt, save_directory)
        #
        #     # L96 time dilation and processing
        #     scaling_time_L96 = dilation_factor * 22.0
        #     L96_obj = externally_provided_currents.L96_object(scaling_time_factor=scaling_time_L96)
        #     L96_x1 = (L96_obj.prepare_f(t_arr).T)[0]
        #     L96_x1[:4000] = 0  # remove transients
        #     process_currents([L96_x1], "L96", ["x1"], dilation_factor, lower_bound, upper_bound, t_arr, dt, save_directory)
        # for upper_bound in [100, 200, 300, 400, 500, 600, 700]:
        upper_bound = 60
        lower_bound = -55
        print("Upper bound: "+str(upper_bound))

        t_initial = -1  # seconds
        t_final = 1  # seconds
        dt = 0.000005  # seconds
        t_arr = np.arange(t_initial, t_final, dt)
        save_directory = "stimuli_May_2023/"

        # L63 time dilation and processing
        scaling_time_L63 = dilation_factor * 22.0
        L63_obj = externally_provided_currents.L63_object(scaling_time_factor=scaling_time_L63)
        I_L63 = L63_obj.prepare_f(t_arr).T
        process_currents(I_L63, "L63", ["x", "y", "z"], dilation_factor, lower_bound, upper_bound, t_arr, dt, save_directory)

        # Colpitts time dilation and processing
        scaling_time_Colpitts = dilation_factor * 150.0
        colp_obj = externally_provided_currents.Colpitts_object(scaling_time_factor=scaling_time_Colpitts)
        I_colpitts = colp_obj.prepare_f(t_arr).T
        process_currents(I_colpitts, "Colpitts", ["x", "y", "z"], dilation_factor, lower_bound, upper_bound, t_arr, dt, save_directory)

        # L96 time dilation and processing
        scaling_time_L96 = dilation_factor * 22.0
        L96_obj = externally_provided_currents.L96_object(scaling_time_factor=scaling_time_L96)
        L96_x1 = (L96_obj.prepare_f(t_arr).T)[0]
        L96_x1[:4000] = 0  # remove transients
        process_currents([L96_x1], "L96", ["x1"], dilation_factor, lower_bound, upper_bound, t_arr, dt, save_directory)


if __name__ == '__main__':
    main()

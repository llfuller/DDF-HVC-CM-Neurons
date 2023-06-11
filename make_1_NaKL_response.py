import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import scipy
import scipy.integrate
import sys
import os
import json
import re
sys.path.append("./data_to_figure_code")
sys.path.append("./making_stimulus_protocols")
import data_loader
import externally_provided_currents as epc
import make_general_plot
#%%
def S0(V,V0,dV0):
    return 0.5*(1.0+np.tanh((V-V0)/dV0))
def tau(V,t0,t1,A,B):
    return t0+t1*(1.0-(np.tanh((V-A)/B))**2)
def g(V,A,B):
    return 0.5*(1.0+np.tanh((V-A)/B))
#%%

def dfdt(t, state, params):
    I = params["I"]
    I_amplitude_scaling = params["I_amplitude_scaling"]
    I_DC = params["I_DC"]
    V, m, h, n = state
    dVdt = gN*m**3*h*(vNa-V) + gK*n**4*(vK-V) + gL*(vL-V) + (I_amplitude_scaling)*I(t) + I_DC
    dmdt = (g(V,vm,dvm) - m)/tau(V,tm0,tm1,vm,dvm)
    dhdt = (g(V,vh,dvh) - h)/tau(V,th0,th1,vh,dvh)
    dndt = (g(V,vn,dvn) - n)/tau(V,tn0,tn1,vn,dvn)
    return [dVdt, dmdt, dhdt, dndt]
#%%
t_start = 500
t_stop = 1000 #ms
dt = 0.02 #ms

times_array = np.arange(start=t_start,stop=t_stop, step=dt)

#%%
# Constants
# capacitance (units: nF)
C = 1.0

# conductances and ion channel reversal potentials
# Units below can be compared to this reference: https://www.researchgate.net/figure/Hodgkin-Huxley-model-current-parameters_fig26_258924750
gN = 120.0 # mS/cm^2
vNa = 50.0 # mV
gK = 20.0
vK = -77.0
gL = 0.3
vL = -54.4

# n gating variable constants
vn = -55.0 # inflection voltage for n gating variable
dvn = 30.0
tn0 = 1.0
tn1 = 5.0

# m gating variable constants
vm = -40.0 # inflection voltage for m gating variable
dvm = 15.0
tm0 = 0.1
tm1 = 0.4

# h gating variable constants
vh = -60.0 # inflection voltage for h gating variable
dvh = -15.0
th0 = 1.0
th1 = 7.0


# Call the function to collect file paths and store the result in list_of_all_stimulus_files
# First Set the target directory, required and excluded substrings
directory = "stimuli_May_2023/"
system_name_to_use_list = ["L63_x"]#, "L63_y", "L63_z", "Colpitts_x", "Colpitts_y", "Colpitts_z", "L96_x1"]
for system_name in system_name_to_use_list:
    required_substrings = ["48","3.5",".txt", system_name]#"(-33.33,100)"] # do not remove ".txt" from this list.
    excluded_substrings = ["FPS", "readme", "README"] # Do not remove elements from this list
    print("Current working directory:", os.getcwd())
    list_of_all_stimulus_files = data_loader.collect_txt_files(directory=directory,
                                                               required_substrings=required_substrings,
                                                               excluded_substrings=excluded_substrings)
    print("required_substrings:", required_substrings)
    print("list_of_all_stimulus_files:", list_of_all_stimulus_files)

    # Print the list of file paths
    print(list_of_all_stimulus_files)
    all_dataset_dicts = []
    for filepath_I_stim in list_of_all_stimulus_files:
        loaded_file_tI = np.loadtxt(filepath_I_stim)
        I_stim = loaded_file_tI[:,1]
        times_array_widened = loaded_file_tI[:,0]*1000 # converting from seconds to ms

        # I_scaling_time_factor = 0.3
        I_amplitude_scaling = 1.0
        I_DC = 0

        # L63_obj = epc.L63_object(scaling_time_factor=I_scaling_time_factor)
        # L63_obj.prepare_f(times_array_widened)
        # L63_obj.function(N=3, t=times_array)
        I_time_dilation_factor = data_loader.pull_time_dilation_from_string(filepath_I_stim)

        # I = 40*np.sin(range(times_array_widened.shape[0]))
        I_interp = scipy.interpolate.interp1d(times_array_widened, I_stim)

        # I_interp = L63_obj.interp_function
        params = {"I":I_interp,"I_amplitude_scaling":I_amplitude_scaling, "I_DC":I_DC}

        state_initial = 0.1*np.ones((4))

        # Replace the odeint call with a solve_ivp call
        sol_ivp = scipy.integrate.solve_ivp(dfdt, (times_array[0], times_array[-1]), state_initial, t_eval=times_array,
                                            args=(params,))

        # Extract the solution array (sol.y) and transpose it to get the same shape as before
        sol = sol_ivp.y.T
        print(f"Sol is shape:{sol.shape}")

        plt.figure()
        plt.plot(times_array, sol[:,0])
        plt.title(f"NaKL Response to L63x with Amplitude\_Scaling={I_amplitude_scaling} I\_DC={I_DC} time\_scaling={I_time_dilation_factor}")
        plt.show()

        directory, filename = os.path.split(filepath_I_stim)
        save_folder = "Single_NaKL_Twin_Experiment_Original_data/"
        # Create the directory if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # find system name from file string
        # Define the string patterns to look for
        start_pattern = "stimuli_April_2023/"
        end_pattern = "_time_dilation="
        # Extract the desired substring
        start_index = filename.find(start_pattern) + len(start_pattern)
        end_index = filename.find(end_pattern)
        system_name_with_underscores = filename[start_index:end_index]
        system_name = system_name_with_underscores.replace("_", "\\_")

        filename_1 = "NaKL_1_V(t)_Response_to_"+filename[:-4]+f"_with_Amplitude_Scaling={I_amplitude_scaling}_I_DC={I_DC}"+f"_(t_start, t_stop)={(t_start,t_stop)}"
        make_general_plot.time_series_plot_and_save(sol[:,0], dt, save_folder,
                                                    title=f"NaKL Response to {system_name} with Amplitude\_Scaling={I_amplitude_scaling} I\_DC={I_DC} time\_scaling={I_time_dilation_factor}",
                                                    save_filename=filename_1,
                                                    t_array=times_array/1000, xlabel="Time (s)", ylabel="Voltage (mV)", show_plot = False)
        filename_2 = filename[:-4]+f"_with_Amplitude_Scaling={I_amplitude_scaling}_I_DC={I_DC}"+f"_(t_start, t_stop)={(t_start,t_stop)}"
        make_general_plot.time_series_plot_and_save((I_amplitude_scaling)*I_interp(times_array) + I_DC, dt, save_folder,
                                                    title=f"{system_name} with Amplitude\_Scaling={I_amplitude_scaling} I\_DC={I_DC} time\_scaling={I_time_dilation_factor}",
                                                    save_filename=filename_2,
                                                    t_array=times_array/1000, xlabel="Time (s)", ylabel="Current (pA)", show_plot = False)

        np.savetxt(save_folder+filename_1+".txt", np.column_stack((sol[:, 0], (I_amplitude_scaling)*I_interp(times_array)+ I_DC, times_array)))

        # Create dataset dictionary
        empty_array = np.array([])
        match = re.search(r'I_(\w+?)_time', filename)
        if match:
            I_stim_name = match.group(1)
        else:
            I_stim_name = "unknown"

        dataset_name = f"NaKL_1_V(t)_Response_to_{filename[:-4]}"
        if t_start!=0:
            dataset_name += f"_(t_start, t_stop)={(t_start,t_stop)}"

        dataset_dict = data_loader.create_dataset_dict(
            name=dataset_name,
            V=None,#sol[:, 0],
            I_stim=None,#I_interp(times_array),
            t=None,#times_array,
            dt=dt/1000, # need to convert this to seconds
            neuron_type="NaKL",
            collection_year=2023,
            data_filepath_dict={
                save_folder + "NaKL_1_V(t)_Response_to_" + filename[
                                                           :-4] + f"_with_Amplitude_Scaling={I_amplitude_scaling}_I_DC={I_DC}"+f"_(t_start, t_stop)={(t_start,t_stop)}.txt": [
                    "V", "I_stim", "t"]
            },
            data_filepath_original=filepath_I_stim,
            I_stim_name = I_stim_name,
            extra_notes=f"Driving current range: {I_stim.min()} to {I_stim.max()}, DC current: {I_DC}, Time dilation: {I_time_dilation_factor}, (t_start, t_stop)={(t_start,t_stop)}"
        )

        # Append the dataset dictionary to the list
        all_dataset_dicts.append(dataset_dict)

    # Save the list of dictionaries as a JSON file
    # with open(save_folder+"all_dataset_dicts.json", "w") as outfile:
    #     json.dump(all_dataset_dicts, outfile)
    data_loader.update_json_file(save_folder+"all_dataset_dicts.json", all_dataset_dicts)

    def create_readme(save_folder, filename, V_units, I_units, t_units):
        with open(os.path.join(save_folder, filename), 'w') as f:
            f.write("README for datasets in folder: " + save_folder + "\n\n")
            f.write("Each dataset contains 3 columns with the following units:\n")
            f.write("Column 1: Voltage (V) in " + V_units + "\n")
            f.write("Column 2: Current (I) in " + I_units + "\n")
            f.write("Column 3: Time (t) in " + t_units + "\n")

    create_readme(save_folder, "README.txt", "mV", "pA", "s")


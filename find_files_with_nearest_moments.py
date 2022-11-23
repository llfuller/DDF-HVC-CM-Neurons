import scipy.stats
import sklearn.metrics
import save_utilities
import numpy as np
import copy
import glob

# For each file in a directory, use the current's statistical moments to match to previously generated current arrays with known moments.


def calc_moments_of_files_in_directory(root_directory = "Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/"):
    """
    # Data directory to recursively load data from:
    root_directory = "Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/" ; Include the final "/"
    """
    # specify what the neuron names are in the file titles here:
    neuron_name_list = ['32425a75',
                        '920061fe']  # example: ['32425a75', '920061fe'] are two CM neurons from Meliza's 2014 data
    Current_units = "pA"
    Voltage_units = "mV"
    Time_units = "ms"
    TT = 0.02  # delta t in Time_units units, time between samples if not specified through loaded files

    file_extension = "txt"  # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase

    # Use only this file:
    files_to_evaluate = [
        "epoch_10.txt"]  # "biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively

    do_not_use_list = ["2014_09_10_0001.abf",
                       "2014_09_10_0002.abf",
                       "2014_09_10_0003.abf"
                       # file attached to Meliza's data when I received it said not to use these. The input signals are not suitable for training
                       ]  # bad data for RBF training

    # ======== do not modify below ==========
    print("Extensions searched: " + str(save_utilities.glob_extension_case_string_builder(file_extension)))
    full_paths_list = glob.glob(
        root_directory + "**/*." + str(save_utilities.glob_extension_case_string_builder(file_extension)),
        recursive=True)
    neuron_name = ""  # leave as "" if no neuron name is found
    # Files to ignore within directory:

    extensions_with_included_unit_data = ["abf", "mat"]

    # Code for 2014 Meliza CM data
    for i, path in enumerate(full_paths_list):
        full_paths_list[i] = path.replace("\\", "/")

    print("Full paths list:" + str(full_paths_list))

    # In[4]:
    moments_dict = {}
    for a_path in full_paths_list:
        if file_extension.lower() == "txt":
            if "voltage" in a_path.lower():  # skip files if 'voltage' in filename. Only need to perform rest of this loop when 'current' in filename, to avoid duplicating work.
                continue
        last_slash_location = a_path.rfind("/")
        a_filename = a_path[last_slash_location + 1:]
        if len(files_to_evaluate) > 0 and a_filename not in files_to_evaluate:
            continue
        directory_to_read_input_data = a_path[
                                       :last_slash_location + 1]  # should include the last slash, but nothing past it
        directory_to_store_plots = "plots/" + directory_to_read_input_data + str(a_filename[:-4]) + "/"
        directory_to_store_txt_data = "data_derived/" + directory_to_read_input_data + 'txt_V_I_t/'
        directory_to_store_moments_txt_data = "data_derived/" + directory_to_read_input_data + 'current_statistical_moments/'
        neuron_name = save_utilities.give_name_if_included_in_path(a_path, neuron_name_list)
        print("================================New File ==================================================")
        if a_filename in do_not_use_list:
            continue  # skip this iteration if the filename is on the do_not_use_list
        if file_extension.lower() in extensions_with_included_unit_data:  # primarily .abf and .mat files
            print("File may have included units which will override units specified by user at top of this code.")
            units_list = save_utilities.load_and_prepare_abf_or_mat_data(directory_to_read_input_data, a_filename,
                                                                         directory_to_store_txt_data, file_extension)
            Current_units, Voltage_units, Time_units = units_list
            imported_data = np.loadtxt(directory_to_store_txt_data + str(a_filename[:-4]) + "_VIt.txt")

            loaded_V = imported_data[:, 0]
            loaded_I = imported_data[:, 1]
            loaded_t = imported_data[:, 2]
        else:  # primarily .txt files
            if "Data2022-50KhZ/" in root_directory:
                loaded_IV = np.loadtxt(a_path)
                loaded_I = loaded_IV[:, 0]
                loaded_V = loaded_IV[:, 1]
            else:
                if 'current' in a_path:
                    voltage_filepath = a_path.replace('current', 'voltage')
                    current_filepath = copy.deepcopy(a_path)
                    a_path.replace('current', '')
                if 'Current' in a_path:
                    voltage_filepath = a_path.replace('Current', 'Voltage')
                    current_filepath = copy.deepcopy(a_path)
                    a_path.replace('Current', '')
                loaded_V = np.loadtxt(voltage_filepath)
                loaded_I = np.loadtxt(a_path)
                loaded_I = np.loadtxt(current_filepath)
            loaded_t = TT * np.array(range(len(loaded_V)))

        total_num_timesteps_in_data = len(loaded_V)

        moments = np.zeros(4)
        moments[0] = scipy.stats.moment(a=loaded_I, moment=1)
        moments[1] = scipy.stats.moment(a=loaded_I, moment=2)
        moments[2] = scipy.stats.moment(a=loaded_I, moment=3)
        moments[3] = scipy.stats.moment(a=loaded_I, moment=4)
        moments_path_str = directory_to_store_moments_txt_data + a_filename[:-4]
        moments_dict[moments_path_str] = moments
        print("Average: "+str(np.average(loaded_I)))
        return moments_dict, moments_path_str


def find_nearest_moments(measured_current_array_moments, list_of_generated_current_array_moments):
    mse_arr = np.zeros(np.shape(list_of_generated_current_array_moments))
    for i, generated_moments in enumerate(list_of_generated_current_array_moments):
        mse_arr[i] = sklearn.metrics.mean_squared_error(y_true=measured_current_array_moments, y_pred=generated_moments)
        if i == len(list_of_generated_current_array_moments-1):
            i_best_match = np.argmin(mse_arr)
    return i_best_match


# calculate moments for all currents in directory root_directory
root_directory="Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/"
moments_dict, moments_path_str = calc_moments_of_files_in_directory(root_directory=root_directory)
save_utilities.save_dict_with_makedir(moments_dict, moments_path_str+".npy")
print("Saved moments dict:")
print(moments_dict)
# # Calculate matching moments
# find_nearest_moments(measured_current_array_moments=, list_of_generated_current_array_moments,
#                      list_of_generated_current_array_moments = )
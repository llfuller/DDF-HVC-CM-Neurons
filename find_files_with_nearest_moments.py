import scipy.stats
import sklearn.metrics
import save_utilities
import numpy as np
import copy
import glob

# For each file in a directory, use the current's statistical moments to match to previously generated current arrays with known moments.

def calc_moments_of_files_in_directory(file_extension, root_directory = "Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/"):
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

    # Use only this file:
    files_to_evaluate = []  # "biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively

    do_not_use_list = ["2014_09_10_0001.abf",
                       "2014_09_10_0002.abf",
                       "2014_09_10_0003.abf"
                       # file attached to Meliza's data when I received it said not to use these. The input signals are not suitable for training
                       ]  # bad data for RBF training
    do_not_use_dir_list = ["earlier format and min"] # list of directories to not use.

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
        if "spectrum" in a_path.lower():
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
        for a_dir in do_not_use_dir_list:
            if a_dir in a_path:
                continue
        if file_extension.lower() in extensions_with_included_unit_data:  # primarily .abf and .mat files
            print("File may have included units which will override units specified by user at top of this code.")
            units_list = save_utilities.load_and_prepare_abf_or_mat_data(directory_to_read_input_data, a_filename,
                                                                         directory_to_store_txt_data, file_extension)
            Current_units, Voltage_units, Time_units = units_list
            imported_data = np.loadtxt(directory_to_store_txt_data + str(a_filename[:-4]) + "_VIt.txt")

            loaded_I = imported_data[:, 1]
        else:  # primarily .txt files
            if "Data2022-50KhZ/" in root_directory:
                loaded_IV = np.loadtxt(a_path)
                loaded_I = loaded_IV[:, 0]
            else:
                if 'current' in a_path:
                    voltage_filepath = a_path.replace('current', 'voltage')
                    current_filepath = copy.deepcopy(a_path)
                    a_path.replace('current', '')
                if 'Current' in a_path:
                    voltage_filepath = a_path.replace('Current', 'Voltage')
                    current_filepath = copy.deepcopy(a_path)
                    a_path.replace('Current', '')
                if "making_stimulus_protocols" in a_path:
                    current_filepath = copy.deepcopy(a_path)
                loaded_I = np.loadtxt(current_filepath)
            loaded_t = TT * np.array(range(len(loaded_I)))

        total_num_timesteps_in_data = len(loaded_I)

        moments = np.zeros(5)
        print("Inputting loaded_I with shape: "+str(loaded_I.shape))
        print("attempting to save this to moments[0]:")
        print(scipy.stats.moment(a=loaded_I, moment=2))
        moments[0] = np.mean(loaded_I)
        for i in range(1,len(moments)):
            moments[i] = scipy.stats.moment(a=loaded_I, moment=i+1)
        moments_path_str = directory_to_store_moments_txt_data + a_filename[:-4]
        moments_dict[moments_path_str] = moments
        print("Average: "+str(np.average(loaded_I)))
    print("Returning: "+str((moments_dict, moments_path_str)))
    return [moments_dict, moments_path_str]


# def find_nearest_moments(measured_current_array_moments, list_of_generated_current_array_moments):
#     mse_arr = np.zeros(np.shape(list_of_generated_current_array_moments))
#     for i, generated_moments in enumerate(list_of_generated_current_array_moments):
#         mse_arr[i] = sklearn.metrics.mean_squared_error(y_true=measured_current_array_moments, y_pred=generated_moments)
#         if i == len(list_of_generated_current_array_moments-1):
#             i_best_match = np.argmin(mse_arr)
#     return i_best_match

def find_nearest_moments(arr_moments_experimental, dict_generated_current_array):
    """
    For measured current, compare against all simulated currents and return name and MSE array of best match
    """
    lowest_mse_found = 100000000000

    arr_max_gen_current_moment = [0,0,0,0,0]
    # Find max absolute value of each of the separate four moments through all current protocols used, for later normalization
    for key, value in dict_generated_current_array.items():
        for i in range(len(value)):
            if arr_max_gen_current_moment[i] < np.abs(value[i]):
                arr_max_gen_current_moment[i] = np.abs(value[i])

    for key, value in dict_generated_current_array.items():
        generated_name = key
        generated_moments = value
        mse_new = sklearn.metrics.mean_squared_error(y_true=np.divide(arr_moments_experimental,arr_max_gen_current_moment),
                                                     y_pred=np.divide(generated_moments,arr_max_gen_current_moment))

        print("Presently considered MSE is: "+str(mse_new))
        if mse_new < lowest_mse_found:
            print("***Changed closest keys: MSE reduced from "+str(lowest_mse_found)+" to "+str(mse_new)+"***")
            lowest_mse_found = mse_new
            generated_moments_closest = generated_moments
            generated_name_closest = generated_name
        print("Checked through "+str((key, value)))
    return [generated_name_closest, generated_moments_closest]

# calculate and save moments for all experimentally measured currents
print("===============================================================================================================")
print("Calculate and save moments for all experimentally measured currents")
print("===============================================================================================================")

root_directory="Data2022-50KhZ/7-7-2022/"
dict_moments_experimental, moments_path_str = calc_moments_of_files_in_directory(file_extension="txt",
                                                                                root_directory=root_directory)
arr_moments_experimental = list(dict_moments_experimental.values())[0]
save_utilities.save_dict_with_makedir(dict_moments_experimental, "data_derived/current_matching/Data2022-50KhZ/experimental_current_moments.npy")
print("Saved moments dict:")
print(dict_moments_experimental)

# calculate moments for all protocol currents
print("===============================================================================================================")
print("Calculate moments for all protocol currents")
print("===============================================================================================================")

root_directory="making_stimulus_protocols/"
dict_moments_protocol, moments_path_str = calc_moments_of_files_in_directory(file_extension="atf",
                                                                             root_directory=root_directory)
# moments_protocol = list(dict_moments_protocol.values())[0]
save_utilities.save_dict_with_makedir(dict_moments_protocol, "data_derived/current_matching/original_current_moments.npy")
print("Saved moments dict:")
print(dict_moments_protocol)

print("===============================================================================================================")
print("Compare and record matches")
print("===============================================================================================================")


# Make list of all .npy files saved
file_extension = "npy"  # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase
full_paths_list = glob.glob(
    "data_derived/Data2022-50KhZ/7-7-2022/" + "**/*." + str(save_utilities.glob_extension_case_string_builder(file_extension)),
    recursive=True)

# create list linking two pathnames suspected to be the same current
linked_currents_list = []

# For each dictionary stored in this experimental list
for a_path in full_paths_list:
    if "moments" not in a_path.lower():  # skip files if 'voltage' in filename. Only need to perform rest of this loop when 'current' in filename, to avoid duplicating work.
        continue
    last_slash_location = a_path.rfind("/")
    a_filename = a_path[last_slash_location + 1:]
    # Use only this file:
    files_to_evaluate = []  # "biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively

    if len(files_to_evaluate) > 0 and a_filename not in files_to_evaluate:
        continue
    directory_to_read_input_data = a_path[
                                   :last_slash_location + 1]  # should include the last slash, but nothing past it
    # directory_to_store_plots = "plots/" + directory_to_read_input_data + str(a_filename[:-4]) + "/"
    # directory_to_store_txt_data = "data_derived/" + directory_to_read_input_data + 'txt_V_I_t/'
    # directory_to_store_moments_txt_data = "data_derived/" + directory_to_read_input_data + 'current_statistical_moments/'
    # neuron_name = save_utilities.give_name_if_included_in_path(a_path, neuron_name_list)
    print("Attempting to load experimental data:")
    loaded_npy = np.load("data_derived/current_matching/Data2022-50KhZ/experimental_current_moments.npy", allow_pickle=True)
    experimental_current_dict = loaded_npy.item()

    for key, value in experimental_current_dict.items():
        arr_moment_experimental = value
        experimental_current_pathname = key

        # Calculate matching moments and return name associated protocol moments
        name_closest, moments_closest = find_nearest_moments(arr_moments_experimental = arr_moment_experimental,
                                        dict_generated_current_array = dict_moments_protocol)

        print(a_path +" is closest to original "+str(name_closest))
        linked_currents_list.append([experimental_current_pathname,name_closest])
        print("Matched using numbers:\n")
        for key, value in experimental_current_dict.items():
            print((value,moments_closest))

print("Finally, the linked list of currents:\n"+str(linked_currents_list))
# np.savetxt("data_derived/experimental_currents_and_suspected_original_currents.txt", linked_currents_list)
# with open("data_derived/experimental_currents_and_suspected_original_currents.txt", "w") as output:
#     output.write(str(linked_currents_list))

with open("data_derived/current_matching/experimental_currents_and_suspected_original_currents.txt", 'w') as output:
    for row in linked_currents_list:
        output.write(str(row) + '\n')
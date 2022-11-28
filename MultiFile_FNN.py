#!/usr/bin/env python
# coding: utf-8

# In[1]:
import save_utilities
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import fnn
import re
import tqdm
import collections
import os

random.seed(2022)
np.random.seed(2022)

# This code is an edit branching off of Python code
# "MultiFile_FPS_Plot_and_Train_Test_RBF.py"
# This code also contains segments from fnn.py written by Barry Xue, central to this script.
# on Nov 27, 2022. Uses the basic file iteration
# FNN code written by Barry Xue provides the core functionality,
# MultiFile code written by Lawson Fuller helps integrate it into existing file directory systems to make iterating over
# existing data easier.

# In[2]:
# modify this
save_and_or_display = "save"
file_extension = "txt"  # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase

# specify what the neuron names are in the file titles here:
neuron_name_list = [
    "Lilac 114 Neuron 1"
]  # example: ['32425a75', '920061fe'] are two CM neurons from Meliza's 2014 data
Current_units = "pA"
Voltage_units = "mV"
Time_units = "ms"
TT = 0.02  # delta t in Time_units units, time between samples if not specified through loaded files

# Data directory to recursively load data from:
root_directory = "Data2022-50KhZ/7-7-2022/"  # example: "HVC_biocm_data/simulations/" ; Include the final "/"

# Use only this file:
files_to_evaluate = [
    # "epoch_1.txt"
]  # "biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively

do_not_use_list = (
    []  # file attached to Meliza's data when I received it said not to use these. The input signals are not suitable for training
)  # bad data for RBF training

FPS_xlim = (0, 0.175)

fraction_of_data_for_training = 4.0 / 6.0

window = 100000 # number of timesteps to either side to consider when looking for nearest neighbors, ex: 5 => 10 total.
tau = 5
R_ratio = 1e-2
D_arr = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20])
save_data = True




# ======== do not modify below ==========
print(
    "Extensions searched: "
    + str(save_utilities.glob_extension_case_string_builder(file_extension))
)
full_paths_list = glob.glob(
    root_directory
    + "**/*."
    + str(save_utilities.glob_extension_case_string_builder(file_extension)),
    recursive=True,
)
neuron_name = ""  # leave as "" if no neuron name is found
# Files to ignore within directory:

extensions_with_included_unit_data = ["abf", "mat"]

# Code for 2014 Meliza CM data
for i, path in enumerate(full_paths_list):
    full_paths_list[i] = path.replace("\\", "/")

print("Full paths list:" + str(full_paths_list))

# In[4]:

for a_path in full_paths_list:
    if file_extension.lower() == "txt":
        if (
            "voltage" in a_path.lower()
        ):  # skip files if 'voltage' in filename. Only need to perform rest of this loop when 'current' in filename, to avoid duplicating work.
            continue
    last_slash_location = a_path.rfind("/")
    a_filename = a_path[last_slash_location + 1 :]
    if len(files_to_evaluate) > 0 and a_filename not in files_to_evaluate:
        continue
    directory_to_read_input_data = a_path[
        : last_slash_location + 1
    ]  # should include the last slash, but nothing past it
    directory_to_store_plots = ("plots/" + directory_to_read_input_data + str(a_filename[:-4]) + "/")
    directory_to_store_txt_data = ("data_derived/" + directory_to_read_input_data + "txt_V_I_t/")
    directory_to_store_FNN_data = ("data_derived/" + directory_to_read_input_data + "FNN/")

    neuron_name = save_utilities.give_name_if_included_in_path(a_path, neuron_name_list)
    print("================================New File ==================================================")
    if a_filename in do_not_use_list:
        continue  # skip this iteration if the filename is on the do_not_use_list
    if (
        file_extension.lower() in extensions_with_included_unit_data
    ):  # primarily .abf and .mat files
        print(
            "File may have included units which will override units specified by user at top of this code."
        )
        units_list = save_utilities.load_and_prepare_abf_or_mat_data(
            directory_to_read_input_data,
            a_filename,
            directory_to_store_txt_data,
            file_extension,
        )
        Current_units, Voltage_units, Time_units = units_list
        imported_data = np.loadtxt(
            directory_to_store_txt_data + str(a_filename[:-4]) + "_VIt.txt"
        )

        loaded_V = imported_data[:, 0]
        loaded_I = imported_data[:, 1]
        loaded_t = imported_data[:, 2]
    else:  # primarily .txt files
        if "Data2022-50KhZ/" in root_directory:
            loaded_IV = np.loadtxt(a_path)
            loaded_I = loaded_IV[:, 0]
            loaded_V = loaded_IV[:, 1]
        else:
            if "current" in a_path:
                voltage_filepath = a_path.replace("current", "voltage")
            if "Current" in a_path:
                voltage_filepath = a_path.replace("Current", "Voltage")
            loaded_V = np.loadtxt(voltage_filepath)
            loaded_I = np.loadtxt(a_path)
        loaded_t = TT * np.array(range(len(loaded_V)))

    total_num_timesteps_in_data = len(loaded_V)
    print(loaded_I.shape)
    datasets, time_delay_indices, time_delay_datapairs = fnn.run_this(loaded_I, loaded_V, loaded_t,
                                                                      search_window_size = window,
                                                                      tau = tau,
                                                                      R_ratio = R_ratio,
                                                                      D_arr = D_arr,
                                                                      save_data = save_data
                                                                      )

    # datasets is a list of len(D_E_array) arrays, with shape (timesteps, D_E)
    for dataset_index, a_time_delay_dataset in enumerate(datasets):
        print(a_time_delay_dataset.shape)
        print(f"Saving .npy files for D={a_time_delay_dataset.shape[1]}")
        save_utilities.save_npy_with_makedir(time_delay_datapairs[dataset_index], directory_to_store_FNN_data+f"min_datapairs_D={a_time_delay_dataset.shape[1]}_window={window}_datapoints")
        save_utilities.save_npy_with_makedir(time_delay_indices[dataset_index],   directory_to_store_FNN_data+f"min_datapairs_D={a_time_delay_dataset.shape[1]}_window={window}_location")

    #data_derived/" + directory_to_read_input_data +
    """User Defined Parameters: specify root path for loaded files"""
    root = directory_to_store_FNN_data

    """===================================
    Load Saved min distance datasets
    ==================================="""
    # load from all the npy file
    D_data_dict, D_index_dict = {}, {}

    try:
        os.listdir(directory_to_store_FNN_data)
    except:
        root = directory_to_store_FNN_data

    for filepath in os.listdir(root):
        print("found directories:\n")
        print(filepath)
        if 'min_datapairs' not in filepath:
            continue
        # print(filepath)
        D = int(re.search("D=([0-9]*)", filepath).group(1))
        window_size = int(re.search("window=([0-9]*)", filepath).group(1))
        data = True if re.search("datapoints", filepath) else False
        if data:
            D_data_dict[(D, window_size)] = np.load(root+filepath, allow_pickle=True)
        else:
            D_index_dict[(D, window_size)] = np.load(root+filepath, allow_pickle=True)


    # evaluate each dataset's FNN ratio with different R values
    D_results = collections.defaultdict(list)  # list of fnn ratios for each (D, window_size combo)
    exp = np.concatenate(
        [np.arange(5, 1, -1), np.array([0]), np.arange(-1, -5, -1)])  # array([ 5,  4,  3,  2, -1, -2, -3, -4])
    for (d, window), d_data in tqdm.tqdm(D_data_dict.items()):
        for e in exp:
            R = float(f'1e{e}')
            fnn_ratio = fnn.count_fnn(D_index_dict[(d, window)], d_data, threshold_R=R)
            D_results[(d, window)].append(fnn_ratio)


    # divide the graph into two plots by windows, and sort by D values
    D_100000 = {}
    D_1000 = {}

    for (d, window_size), lst in D_results.items():
        if window_size == 1000:
            D_1000[d] = lst
        if window_size == 100000:
            D_100000[d] = lst

    D_1000 = dict(sorted(D_1000.items(), key=lambda x: x[0]))
    D_100000 = dict(sorted(D_100000.items(), key=lambda x: x[0]))

    """
    plot fnn vs R for each D
    """
    window = 100000
    r, c = 5, len(D_100000) // 4
    fig, axes = plt.subplots(r, c, figsize=(18, 14))
    fig.tight_layout(pad=3.0)
    r_i, c_i = 0, 0
    for d, fnn_data in D_100000.items():
        axes[r_i, c_i].set_title(f'D={d}, window={window}')
        axes[r_i, c_i].set_xlabel("R's Exponents")
        axes[r_i, c_i].set_ylabel("FNN Ratio")
        axes[r_i, c_i].scatter(exp, fnn_data)
        c_i += 1
        if c_i == c:
            c_i = 0
            r_i += 1


    data = []
    for (d, window), fnn_data in D_results.items():
        data.append((d, fnn_data[4]))  # get fnn for R's exponent is -1 for all D values

    # extra sorting step because the order for D is weird
    data = sorted(data, key=lambda x: x[0])

    D_1000, D_100000 = [], []
    fnn_lst_1000, fnn_lst_100000 = [], []

    for i in range(len(data)):
        # if i % 2 == 0:
        #     D_1000.append(data[i][0])
        #     fnn_lst_1000.append(data[i][1])
        # else:
        D_100000.append(data[i][0])
        fnn_lst_100000.append(data[i][1])

    """
    Save plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.set_title("FNN Ratio vs D; R=0.1; window=100000")
    plt.scatter(D_100000, fnn_lst_100000, c='green')
    save_utilities.save_fig_with_makedir(figure=fig,
                                         save_location=f"{directory_to_store_plots}FNN/FNN_vs_D_R={R},"
                                                       f"window={window}.png")
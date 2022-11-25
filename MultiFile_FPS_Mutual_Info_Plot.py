#!/usr/bin/env python
# coding: utf-8

# In[1]:
from TimeDelay_Neuron_DDF_GaussianForm import *
import Fourier_Power_Spectrum
import plotting_utilities
import save_utilities
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import copy
import glob
from scipy.ndimage import gaussian_filter1d

import numpy.fft
import math

import os

# random.seed(2022)
# np.random.seed(2022)

# This code is an edit branching off of Python code
# AMI code written by both Randall Clark and Lawson Fuller. Using code from Randall's project.

# In[2]:
# modify this
save_and_or_display = "save"


def AveMutInf(Data1, Data2, lower, upper, nobins):
    PDistA = np.histogram(Data1, nobins, (lower, upper))[0] / len(Data1)
    PDistB = np.histogram(Data2, nobins, (lower, upper))[0] / len(Data2)
    PDistAB = np.histogram2d(Data1, Data2, nobins, ((lower, upper), (lower, upper)))[0] / len(Data1)
    #     Equivalent to:
    #     AveMI2 = 0
    #     for a in range(nobins):
    #         for b in range(nobins):
    #             if PDistA[a] != 0 and PDistB[b] != 0 and PDistAB[a,b] !=0:
    #                 AveMI2 += PDistAB[a,b]*np.log2(PDistAB[a,b]/(PDistA[a]*PDistB[b]))
    #     print(np.sum(AveMI-AveMI2))

    AveMI = np.sum(
        np.ma.masked_invalid(np.multiply(PDistAB, np.log2(np.divide(PDistAB, np.outer(PDistA, PDistB))))))
    return AveMI


def AveMutIGraph(DelayMax,data,length,lower,upper,nobins):
    """
    DelayMax (int): tau ranges from 0 to DelayMax
    data: Data from which you will take pairs time-delayed points and calculate AMI
    length: data is split into two arrays, one from 0 to length, another from tau to length+tau.
            Make sure length + DelayMax < len(data)
    lower: lower bound of histogram < lowest scalar value in data
    upper: upper bound of histogram > highest scalar value in data
    nobins: number of bins in histogram (granularity of probability calculation)
    """
    AveMutIDat = np.zeros(DelayMax)
    for i in range(DelayMax):
        tau = i
        Data1 = data[0:length]
        Data2 = data[tau:length+tau]
        AveMutIDat[i] = AveMutInf(Data1,Data2,lower,upper,nobins)
    return AveMutIDat


# for dataset_id in range():
#
#     # Tau Testing
#     varying_hyperparam = "tau"
#     # Plotting for L63x t-dilation=0.5-driven Red 171 Neuron 2 (Epoch 1).
#     # Data directory to recursively load data from:
#     root_directory = "Data2022-50KhZ/7-7-2022/Red 171/Neuron 2/"  # "biocm_simulations/"#"cm_ddf/data/"#"Data2022-50KhZ/" # example: "HVC_biocm_data/simulations/" ; Include the final "/"
#     files_to_evaluate = ["epoch_1.txt"]  # "biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively
#
#     # Plotting for L63x t-dilation=0.5-driven Lilac 114 Neuron 1 (Epoch 3).
#     root_directory = "Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/"  # "biocm_simulations/"#"cm_ddf/data/"#"Data2022-50KhZ/" # example: "HVC_biocm_data/simulations/" ; Include the final "/"
#     files_to_evaluate = ["epoch_3.txt"]  # "biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively
#
#     # Plotting for L63x t-dilation=0.2-driven Lilac 114 Neuron 2 (Epoch 2).
#     root_directory = "Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 2/"  # "biocm_simulations/"#"cm_ddf/data/"#"Data2022-50KhZ/" # example: "HVC_biocm_data/simulations/" ; Include the final "/"
#     files_to_evaluate = ["epoch_2.txt"]  # "biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively
#
#     # Plotting for L63x t-dilation=0.5-driven Lilac 114 Neuron 1 (Epoch 2).
#
#     # Plotting for L63x t-dilation=0.5-driven Lilac 114 Neuron 1 (Epoch 5).
#
#     # Plotting for L63x t-dilation=0.2-driven Lilac 242 Neuron 1 (Epoch 1).
#
#     # Plotting for L63x t-dilation=0.2-driven Lilac 242 Neuron 1 (Epoch 9).
#
#     # Plotting for L63x t-dilation=0.2-driven Lilac 242 Neuron 1 (Epoch 10).

root_directory = "Data2022-50KhZ/7-7-2022/"#Red 171/Neuron 2/"
file_extension = "txt" # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase
# Use only this file:
files_to_evaluate = []#"epoch_1.txt"]#"biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively


# specify what the neuron names are in the file titles here:
neuron_name_list = ['32425a75',
                     '920061fe'] # example: ['32425a75', '920061fe'] are two CM neurons from Meliza's 2014 data
Current_units = "pA"
Voltage_units = "mV"
Time_units = "ms"
TT = 0.02 # delta t in Time_units units, time between samples if not specified through loaded files

do_not_use_list = ["2014_09_10_0001.abf",
                   "2014_09_10_0002.abf",
                   "2014_09_10_0003.abf"# file attached to Meliza's data when I received it said not to use these. The input signals are not suitable for training
                   ] # bad data for RBF training


# In[3]:

# ======== do not modify below ==========
print("Extensions searched: "+str(save_utilities.glob_extension_case_string_builder(file_extension)))
full_paths_list = glob.glob(root_directory+"**/*."+str(save_utilities.glob_extension_case_string_builder(file_extension)),
                            recursive=True)
neuron_name = "" # leave as "" if no neuron name is found
# Files to ignore within directory:

extensions_with_included_unit_data = ["abf","mat"]

# Code for 2014 Meliza CM data
for i, path in enumerate(full_paths_list):
    full_paths_list[i] = path.replace("\\","/")

print("Full paths list:"+str(full_paths_list))

# In[4]:

for a_path in full_paths_list:
    if file_extension.lower() == "txt":
        if "voltage" in a_path.lower(): # skip files if 'voltage' in filename. Only need to perform rest of this loop when 'current' in filename, to avoid duplicating work.
            continue
    last_slash_location = a_path.rfind("/")
    a_filename = a_path[last_slash_location+1:]
    if len(files_to_evaluate)>0 and a_filename not in files_to_evaluate:
        continue
    directory_to_read_input_data = a_path[:last_slash_location+1] # should include the last slash, but nothing past it
    directory_to_store_plots = "plots/" + directory_to_read_input_data + str(a_filename[:-4]) + "/"
    directory_to_store_txt_data = "data_derived/" + directory_to_read_input_data + 'txt_V_I_t/'
    neuron_name = save_utilities.give_name_if_included_in_path(a_path, neuron_name_list)
    print("================================New File ==================================================")
    if a_filename in do_not_use_list:
        continue # skip this iteration if the filename is on the do_not_use_list
    if file_extension.lower() in extensions_with_included_unit_data: # primarily .abf and .mat files
        print("File may have included units which will override units specified by user at top of this code.")
        units_list = save_utilities.load_and_prepare_abf_or_mat_data(directory_to_read_input_data, a_filename,
                                                        directory_to_store_txt_data, file_extension)
        Current_units, Voltage_units, Time_units = units_list
        imported_data = np.loadtxt(directory_to_store_txt_data + str(a_filename[:-4]) + "_VIt.txt")

        loaded_V = imported_data[:, 0]
        loaded_I = imported_data[:, 1]
        loaded_t = imported_data[:, 2]
    else: # primarily .txt files
        if "Data2022-50KhZ/" in root_directory:
            loaded_IV = np.loadtxt(a_path)
            loaded_I = loaded_IV[:, 0]
            loaded_V = loaded_IV[:, 1]
        else:
            if 'current' in a_path:
                voltage_filepath = a_path.replace('current','voltage')
                current_filepath = copy.deepcopy(a_path)
                a_path.replace('current', '')
            if 'Current' in a_path:
                voltage_filepath = a_path.replace('Current','Voltage')
                current_filepath = copy.deepcopy(a_path)
                a_path.replace('Current','')
            loaded_V = np.loadtxt(voltage_filepath)
            loaded_I = np.loadtxt(a_path)
            loaded_I = np.loadtxt(current_filepath)
        loaded_t = TT*np.array(range(len(loaded_V)))

    fraction_of_data_for_training = 0.4/0.6
    total_num_timesteps_in_data = len(loaded_V)
    train_timestep_end = round(total_num_timesteps_in_data*fraction_of_data_for_training) #4/6 for neuron 2 epoch 5, and 5/6 for everything else
    Voltage_train = loaded_V[:train_timestep_end]
    Current_train = loaded_I[:train_timestep_end]
    Time_train    = loaded_t[:train_timestep_end]
    Voltage_test = loaded_V[train_timestep_end:total_num_timesteps_in_data]
    Current_test = loaded_I[train_timestep_end:total_num_timesteps_in_data]
    Time_test    = loaded_t[train_timestep_end:total_num_timesteps_in_data]
    length = Voltage_train.shape[0]-1000 # - 1000 just to give breathing room
    PreLength = Voltage_test.shape[0]-1000 # - 1000 just to give breathing room


    # In[5]:
    # ===============================  POWER SPECTRA  =====================================
    # FFT Train
    if Time_units.lower() == "s":
        freq_units = "Hz"  # frequency units
    if Time_units.lower() == "ms":
        freq_units = "kHz"  # frequency units
    TT = float(loaded_t[2]-loaded_t[1]) # delta t; time between data samples
    sampling_rate = 1.0/TT # frequency (1/ms = kHz)
    FPS_list, freq_array = Fourier_Power_Spectrum.calc_Fourier_power_spectrum([Current_train, Voltage_train], Time_train)
    delta_freq = freq_array[3]-freq_array[2]
    freq_without_0_index = freq_array[1:]
    power_spectrum = FPS_list[0]
    normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))

    # ===============================  Plotting training and testing current and voltage  =====================================
    plotting_utilities.plotting_quantity(x_arr = Time_train, y_arr = Current_train, title = "Training Current",
                                         xlabel = "Time ("+str(Time_units)+")", ylabel = "Current ("+str(Current_units)+")",
                                         save_and_or_display=save_and_or_display,
                                         save_location=directory_to_store_plots+"I_V_training_and_testing/"+"Train 1 first half Test 1 second half"+" Training Current.png")
    plotting_utilities.plotting_quantity(x_arr = Time_train, y_arr = Voltage_train, title = "Training Voltage",
                                         xlabel = "Time ("+str(Time_units)+")", ylabel = "Voltage ("+str(Voltage_units)+")",
                                         save_and_or_display=save_and_or_display,
                                         save_location=directory_to_store_plots+"I_V_training_and_testing/"+"Train 1 first half Test 1 second half"+" Training Voltage.png")

    plotting_utilities.plotting_quantity(x_arr = Time_test, y_arr = Current_test, title = "Test Current",
                                         xlabel = "Time ("+str(Time_units)+")", ylabel = "Current ("+str(Current_units)+")",
                                         save_and_or_display=save_and_or_display,
                                         save_location=directory_to_store_plots+"I_V_training_and_testing/"+"Train 1 first half Test 1 second half"+" Test Current.png")
    plotting_utilities.plotting_quantity(x_arr = Time_test, y_arr = Voltage_test, title = "Test Voltage",
                                         xlabel = "Time ("+str(Time_units)+")", ylabel = "Voltage ("+str(Voltage_units)+")",
                                         save_and_or_display=save_and_or_display,
                                         save_location=directory_to_store_plots+"I_V_training_and_testing/"+"Train 1 first half Test 1 second half"+" Test Voltage.png")

    AMI_plots_save_location = directory_to_store_plots + "Avg_Mut_Info/"
    #Make save directory if it doesn't exist:
    if "/" in AMI_plots_save_location:
        last_slash_index = AMI_plots_save_location.rfind('/') #finds last location of "/" in save_location
    directory = AMI_plots_save_location[:last_slash_index]
    filename  = AMI_plots_save_location[last_slash_index:]

    exp_data = loaded_V
    exp_data_I = loaded_I
    DelayMax = 70000

    start_time = time.time()
    nobins = 400
    # AMIT = AveMutIGraph(30000,NaKL[0],15000,-160,60,nobins=nobins)
    # plt.figure()
    # plt.plot(exp_data)
    # plt.plot(exp_data_I)
    # plt.show()
    AMI_length = 3000
    print("exp_data.shape" + str(exp_data.shape))
    AMIT = AveMutIGraph(DelayMax=DelayMax,data=exp_data,length=AMI_length,lower=-160,upper=60,nobins=nobins)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(AMIT)
    plt.figure(figsize=(20, 10))
    plt.plot(np.multiply(TT,np.array(range(DelayMax))),AMIT, label='Average Mutual Information')
    plt.xlabel("tau ("+str(Time_units)+")")
    plt.ylabel("Average Mutual Information")
    # plt.xlim(0, 1000)
    # plt.ylim(0, 0.1)
    plt.legend()
    # plt.show()
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if os.path.isdir(directory):
        plt.savefig(directory +"/"+ str(a_filename[:-4])+"png", bbox_inches='tight')
    plt.close("all")

    if not os.path.isdir("data_derived/" + directory_to_read_input_data +"/Avg_Mut_Info/"):
        os.makedirs("data_derived/" + directory_to_read_input_data +"/Avg_Mut_Info/")
    if os.path.isdir("data_derived/" + directory_to_read_input_data +"/Avg_Mut_Info/"):
        np.savetxt("data_derived/" + directory_to_read_input_data +"/Avg_Mut_Info/"+ str(a_filename[:-4])+"png",
                   np.hstack((np.multiply(TT,np.array(range(DelayMax))), AMIT)))

    # No need to train or predict (other scripts take care of that)


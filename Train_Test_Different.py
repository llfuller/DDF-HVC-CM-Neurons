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
# "MultiFile_FPS_Plot_and_Train_Test_RBF"
# on Nov 29 ish. The purpose of this file is to train on one epoch and test on another.

# In[2]:
# modify this
save_and_or_display = "save"

# Tau Testing
varying_hyperparam = "tau"

""" Parameters for 	
*Red 171 Neuron 2 Epoch 1:
		_epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10
			driven by "Current vs Time (I_colpitts_x_time_dilation=0.5).txt
"""
# Testing MSE convolutional sigma with hyperparams that worked for L63x t-dilation=0.2-driven Lilac 242 Neuron 1 (Epoch 10).
tau_arr = np.array([10])#np.array(range(10, 20)) # math notation: range(2,10) = all integers in bounds [2,9)
D_arr = np.array([10])#np.array(range(2, 10)) # math notation: range(2,10) = all integers in bounds [2,9)
beta_arr = np.array(np.power(10.0,[-3]))#np.array(np.power(10.0,range(-3,3))) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
R_arr = np.array(np.power(10.0,[-3])) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
file_extension = "txt" # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase


# specify what the neuron names are in the file titles here:
neuron_name_list = ['32425a75',
                     '920061fe'] # example: ['32425a75', '920061fe'] are two CM neurons from Meliza's 2014 data
Current_units_train = "pA"
Voltage_units_train = "mV"
Time_units_train = "ms"
Current_units_test = "pA"
Voltage_units_test = "mV"
Time_units_test = "ms"

TT = 0.02 # delta t in Time_units units, time between samples if not specified through loaded files

# Data directory to recursively load data from:
root_directory = "Data2022-50KhZ/7-7-2022/Red 171/Neuron 2/"#"biocm_simulations/"#"cm_ddf/data/"#"Data2022-50KhZ/" # example: "HVC_biocm_data/simulations/" ; Include the final "/"

# Use only this file:
files_to_evaluate = ["epoch_1.txt"]#"biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively

do_not_use_list = ["2014_09_10_0001.abf",
                   "2014_09_10_0002.abf",
                   "2014_09_10_0003.abf"# file attached to Meliza's data when I received it said not to use these. The input signals are not suitable for training
                   ] # bad data for RBF training

neuron_number = "Neuron 2"
neuron_data_path   = f"Data2022-50KhZ/7-7-2022/Red 171/{neuron_number}/"
training_data_path = neuron_data_path+"epoch_1.txt"
testing_data_path  = neuron_data_path+"epoch_6.txt"

FPS_xlim= (0,0.175)

fraction_of_data_for_training = 4.0/6.0

sigma_c_list = np.array([0, 5, 10, 100, 500, 1000, 2000, 5000]) # list of convolutional gaussian sigma for MSE calc
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

def load_VIt_from_file(file_path, neuron_data_path, Current_units, Voltage_units, Time_units):
    last_slash_location = file_path.rfind("/")
    a_filename = file_path[last_slash_location + 1:]
    units_list = Current_units, Voltage_units, Time_units # default values as input
    directory_to_read_input_data = file_path[
                                   :last_slash_location + 1]  # should include the last slash, but nothing past it
    directory_to_store_plots = "plots/" + neuron_data_path + str(a_filename[:-4]) + "/"
    directory_to_store_txt_data = "data_derived/" + neuron_data_path + 'txt_V_I_t/'
    neuron_name = save_utilities.give_name_if_included_in_path(file_path, neuron_name_list)
    print("================================New File ==================================================")

    if file_extension.lower() in extensions_with_included_unit_data: # primarily .abf and .mat files
        print("File may have included units which will override units specified by user at top of this code.")
        units_list = save_utilities.load_and_prepare_abf_or_mat_data(directory_to_read_input_data, a_filename,
                                                        directory_to_store_txt_data, file_extension)
        # Current_units, Voltage_units, Time_units = units_list
        imported_data = np.loadtxt(directory_to_store_txt_data + str(a_filename[:-4]) + "_VIt.txt")

        loaded_V = imported_data[:, 0]
        loaded_I = imported_data[:, 1]
        loaded_t = imported_data[:, 2]
    else: # primarily .txt files
        if "Data2022-50KhZ/" in root_directory:
            loaded_IV = np.loadtxt(file_path)
            loaded_I = loaded_IV[:, 0]
            loaded_V = loaded_IV[:, 1]
        else:
            if 'current' in file_path:
                voltage_filepath = file_path.replace('current','voltage')
                current_filepath = copy.deepcopy(file_path)
                file_path.replace('current', '')
            if 'Current' in file_path:
                voltage_filepath = file_path.replace('Current','Voltage')
                current_filepath = copy.deepcopy(file_path)
                file_path.replace('Current','')
            loaded_V = np.loadtxt(voltage_filepath)
            loaded_I = np.loadtxt(current_filepath)
        loaded_t = TT*np.array(range(len(loaded_V)))
    return loaded_V, loaded_I, loaded_t, units_list, directory_to_read_input_data, a_filename, directory_to_store_plots, directory_to_store_txt_data, neuron_name


# load VIt from training data
loaded_V_train, loaded_I_train, loaded_t_train, units_list_train, directory_to_read_input_data_train, a_filename_train\
    , directory_to_store_plots_train, directory_to_store_txt_data, neuron_name = load_VIt_from_file(training_data_path, neuron_data_path, Current_units_train, Voltage_units_train, Time_units_train)
Current_units_train, Voltage_units_train, Time_units_train = units_list_train
total_num_timesteps_in_data_train = len(loaded_V_train)
train_timestep_end = round(total_num_timesteps_in_data_train*fraction_of_data_for_training) #4/6 for neuron 2 epoch 5, and 5/6 for everything else
Voltage_train = loaded_V_train[:train_timestep_end]
Current_train = loaded_I_train[:train_timestep_end]
Time_train    = loaded_t_train[:train_timestep_end]

# load VIt from training data
loaded_V_test, loaded_I_test, loaded_t_test, units_list_test, directory_to_read_input_data_test, a_filename_test\
    , directory_to_store_plots_test, directory_to_store_txt_data, neuron_name = load_VIt_from_file(testing_data_path, neuron_data_path, Current_units_test, Voltage_units_test, Time_units_test)
Current_units_test, Voltage_units_test, Time_units_test = units_list_test
total_num_timesteps_in_data_test = len(loaded_V_test)
test_timestep_start = 1000
Voltage_test = loaded_V_test[test_timestep_start:total_num_timesteps_in_data_test]
Current_test = loaded_I_test[test_timestep_start:total_num_timesteps_in_data_test]
Time_test    = loaded_t_test[test_timestep_start:total_num_timesteps_in_data_test]

plt.figure()
plt.plot(Current_train)
plt.plot(Current_test)
plt.title("Current train vs current test")
plt.show()

directory_to_store_plots  = "plots/" + neuron_data_path + "trained_on_" + a_filename_train[:-4] + "_tested_on_" + a_filename_test[:-4] + "/"
directory_to_store_txt_data = "data_derived/" + neuron_data_path + 'txt_V_I_t/' + "trained_on_" + a_filename_train[:-4] + "_tested_on_" + a_filename_test[:-4] + "/"

assert (Current_units_train.lower() == Current_units_train.lower())
assert (Voltage_units_train.lower() == Voltage_units_train.lower())
assert (Time_units_train.lower()    == Time_units_train.lower())
# If these assert statements pass, then we can assign the units generically:
Current_units = Current_units_train
Voltage_units = Voltage_units_train
Time_units    = Time_units_train

length = Voltage_train.shape[0]-1000 # - 1000 just to give breathing room
PreLength = Voltage_test.shape[0]-1000 # - 1000 just to give breathing room

# In[5]:
# ===============================  POWER SPECTRA  =====================================
# FFT Train
if Time_units.lower() == "s":
    freq_units = "Hz"  # frequency units
if Time_units.lower() == "ms":
    freq_units = "kHz"  # frequency units
TT = float(loaded_t_train[2]-loaded_t_train[1]) # delta t; time between data samples
assert (float(loaded_t_train[2]-loaded_t_train[1]) == float(loaded_t_test[2]-loaded_t_test[1]))
sampling_rate = 1.0/TT # frequency (1/ms = kHz)
FPS_list, freq_array = Fourier_Power_Spectrum.calc_Fourier_power_spectrum([Current_train, Voltage_train], Time_train)
delta_freq = freq_array[3]-freq_array[2]
freq_without_0_index = freq_array[1:]
power_spectrum = FPS_list[0]
normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))
#--------------- Current FPS Plots --------------
# Training Current with no modifications
# fig = plotting_utilities.plotting_quantity(x_arr = freq_array, y_arr = power_spectrum/np.max(np.abs(power_spectrum)),
#                                            title = "Power(freq) of current_train (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")",
#                                            xlabel = "Frequency (kHz)",
#                                            ylabel = "Normalized Power (1.0 = max from whole spectrum)",
#                                            save_and_or_display= save_and_or_display,
#                                            save_location=directory_to_store_plots+"Fourier_analysis/"+"Power spectrum of training current (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")_full_data.png",
#                                            xlim=None)
# save_utilities.save_text(data = np.column_stack((freq_array, power_spectrum)),
#                          a_str = save_and_or_display,
#                          save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_current-full_data.txt")


# Training Current - No index 0, Normalized, many windows
for window_size in [0.175]:#5, 75, 150, 300]:
    final_index = int(round(float(window_size) / delta_freq))
    # fig = plotting_utilities.plotting_quantity(x_arr = freq_without_0_index, y_arr = normalized_power_spec_without_0_index,
    #                                            title = "Power(freq) of current_train (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")",
    #                                            xlabel = "Frequency (kHz)",
    #                                            ylabel = "Normalized Power (1.0 = max from [1:])",
    #                                            save_and_or_display= save_and_or_display,
    #                                            save_location=directory_to_store_plots+"Fourier_analysis/"+"Power spectrum of training current (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")_last_freq_shown="+str(window_size)+".png",
    #                                            xlim=(0, window_size))
    # save_utilities.save_text(data = np.column_stack((freq_without_0_index[:final_index], normalized_power_spec_without_0_index[:final_index])),
    #                          a_str = save_and_or_display,
    #                          save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_current-_last_freq_shown="+str(window_size)+".txt")


# --------------- Voltage FPS Plots --------------
power_spectrum = FPS_list[1]
# frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))


# Training Voltage with no modifications
# fig = plotting_utilities.plotting_quantity(x_arr = freq_array, y_arr = power_spectrum/np.max(np.abs(power_spectrum)),
#                                            title = "Power(freq) of voltage_train (Neuron " + str(neuron_name) + '_' + str(a_filename[:-4]) + ")",
#                                            xlabel = "Frequency (kHz)",
#                                            ylabel = "Normalized Power (1.0 = max from whole spectrum)",
#                                            save_and_or_display= save_and_or_display,
#                                            save_location=directory_to_store_plots +"Fourier_analysis/"+ "Power spectrum of training voltage (Neuron " + str(neuron_name) + '_' + str(a_filename[:-4]) + ")_full_data.png",
#                                            xlim=None)
# save_utilities.save_text(data = np.column_stack((freq_array, power_spectrum)),
#                          a_str = save_and_or_display,
#                          save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_voltage-full_data.txt")


# Training Current - No index 0, Normalized, many windows
for window_size in [0.175]:#5, 75, 150, 300]:
    final_index = int(round(float(window_size) / delta_freq))
    # fig = plotting_utilities.plotting_quantity(x_arr = freq_without_0_index, y_arr = normalized_power_spec_without_0_index,
    #                                            title = "Power(freq) of voltage_train (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")",
    #                                            xlabel = "Frequency (kHz)",
    #                                            ylabel = "Normalized Power (1.0 = max from [1:])",
    #                                            save_and_or_display= save_and_or_display,
    #                                            save_location=directory_to_store_plots+"Fourier_analysis/"+"Power spectrum of training voltage (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")_last_freq_shown="+str(window_size)+".png",
    #                                            xlim=(0, window_size))
    # save_utilities.save_text(data = np.column_stack((freq_without_0_index[:final_index], normalized_power_spec_without_0_index[:final_index])),
    #                          a_str = save_and_or_display,
    #                          save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_voltage-_last_freq_shown="+str(window_size)+".txt")

# ===============================  END OF POWER SPECTRA  =====================================
# ===============================  Plotting training and testing current and voltage  =====================================
plotting_utilities.plotting_quantity(x_arr = Time_train, y_arr = Current_train, title = "Training Current",
                                     xlabel = "Time ("+str(Time_units)+")", ylabel = "Current ("+str(Current_units)+")",
                                     save_and_or_display=save_and_or_display,
                                     save_location=directory_to_store_plots+"I_V_training_and_testing/"+a_filename_train[:-4]+" Training Current.png")
plotting_utilities.plotting_quantity(x_arr = Time_train, y_arr = Voltage_train, title = "Training Voltage",
                                     xlabel = "Time ("+str(Time_units)+")", ylabel = "Voltage ("+str(Voltage_units)+")",
                                     save_and_or_display=save_and_or_display,
                                     save_location=directory_to_store_plots+"I_V_training_and_testing/"+a_filename_train[:-4]+" Training Voltage.png")

plotting_utilities.plotting_quantity(x_arr = Time_test, y_arr = Current_test, title = "Test Current",
                                     xlabel = "Time ("+str(Time_units)+")", ylabel = "Current ("+str(Current_units)+")",
                                     save_and_or_display=save_and_or_display,
                                     save_location=directory_to_store_plots+"I_V_training_and_testing/"+a_filename_test[:-4]+" Test Current.png")
plotting_utilities.plotting_quantity(x_arr = Time_test, y_arr = Voltage_test, title = "Test Voltage",
                                     xlabel = "Time ("+str(Time_units)+")", ylabel = "Voltage ("+str(Voltage_units)+")",
                                     save_and_or_display=save_and_or_display,
                                     save_location=directory_to_store_plots+"I_V_training_and_testing/"+a_filename_test[:-4]+" Test Voltage.png")

# Save accompanying text files:
save_utilities.save_text(data = np.column_stack((Time_train, Current_train)),
                         a_str = save_and_or_display,
                         save_location = directory_to_store_txt_data + "I_V_training_and_testing/"+a_filename_train[:-4]+" Training Current.txt")
# Save accompanying text files:
save_utilities.save_text(data = np.column_stack((Time_train, Voltage_train)),
                         a_str = save_and_or_display,
                         save_location = directory_to_store_txt_data + "I_V_training_and_testing/"+a_filename_train[:-4]+" Training Voltage.txt")
# Save accompanying text files:
save_utilities.save_text(data = np.column_stack((Time_test, Current_test)),
                         a_str = save_and_or_display,
                         save_location = directory_to_store_txt_data + "I_V_training_and_testing/"+a_filename_test[:-4]+" Test Current.txt")
# Save accompanying text files:
save_utilities.save_text(data = np.column_stack((Time_test, Voltage_test)),
                         a_str = save_and_or_display,
                         save_location = directory_to_store_txt_data + "I_V_training_and_testing/"+a_filename_test[:-4]+"  Test Phase (True) Voltage.txt")


# In[6]:
num_trials = 1
conv_MSE_trials_array = np.zeros((np.shape(tau_arr)[0], num_trials, sigma_c_list.shape[0]))  # scalar MSE for each R (row) and trial (column)
print("Shape of conv_MSE_trials_array: " +str(np.shape(conv_MSE_trials_array)))
# =============================== Training and Prediction  =====================================
print(f"Beginning Training on {training_data_path} and prediction for {testing_data_path}")
for trial_index in range(num_trials):
    trial_number_string = "trial_" + str(trial_index)
    for tau_index, tau in enumerate(tau_arr):
        for D_index, D in enumerate(D_arr):
            # random.seed(2022)
            # np.random.seed(2022)

            print("========================New tau and D combination ==================")
            # if tau != tau_specified or D!= D_specified:
            #     continue  # want to try predicting only this neuron since it's complex and most interesting
            time_start = time.time()
            Xdata = Voltage_train
            NoCenters_no_thresh = 500
            # NoCenters_above_thresh = 50
            DDF = Gauss()
            # Combine centers above threshold with centers determined by kmeans
            Centers_k_means = DDF.KmeanCenter(Xdata,NoCenters_no_thresh,D,length,tau);
            time_k_centers_done = time.time()
            print("Time to find k centers: "+str(time_k_centers_done-time_start))
            # temp_array = copy.deepcopy(Xdata)
            # temp_array[temp_array<-50]=-100
            # Centers_above_thresh = DDF.KmeanCenter(temp_array,NoCenters_above_thresh,D,length,tau);
            # Center = np.concatenate((Centers_k_means,Centers_above_thresh),axis=0)
            Center = Centers_k_means

            NoCenters = np.shape(Center)[0]
            print(NoCenters)
            print("Centers:"+str(Center.shape))
            # np.savetxt('centers/Center '+str(neuron_name)+'_'+str(a_filename[:-4])+'(D,tau,NumCenters)='+str((D,tau,NoCenters))+'.txt',Center)
            Center = np.loadtxt('centers/Center '+str(neuron_name)+'_'+str(a_filename_train[:-4])+'(D,tau,NumCenters)='+str((D,tau,NoCenters))+'.txt')
            print("Centers used: \n "+str(Center[:6,0]))


            stim_train = Current_train
            Pdata = Voltage_test
            bias = tau*(D-1)+1#50 # should be larger than tau*(D-1) or something like that
            # X = np.arange(bias,bias+PreLength*TT,TT)
            X = Time_test[bias:bias+PreLength]#bias -1 maybe?

            time_preparing_to_run_beta_r = time.time()
            print("Time to reach right before beta_r loop: "+str(time_preparing_to_run_beta_r-time_start))
            # # In[7]:
            for beta in beta_arr:
                for R_index, R in enumerate(R_arr):
                    # if (not math.isclose(beta,1.0)) or (not math.isclose(R,0.01)):
                    #     continue
                    print("(tau, D, beta, R) = " + str((tau, D, beta,R)))
                    time_beta_r_start = time.time()
                    title = "Train test different "+str(neuron_name)+'_'+neuron_number+' with tstep='+str(TT)+' '+str(Time_units)+', D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train TSteps = '+str(length)+', Centers = '+str(NoCenters)+', tau = '+str(tau)
                    # print(R)
                    # print("Shape of Xdata is now "+str(Xdata.shape))
                    # plt.figure()
                    # plt.plot(Xdata)
                    # plt.plot(stim_train)
                    # plt.title("Training Stimulus and Voltage")
                    # plt.xlabel("Time")
                    # plt.ylabel("Voltage or Current")
                    # plt.show()
                    F = DDF.FuncApproxF(Xdata,length,Center,beta,R,D,stim_train,tau)
                    time_beta_r_trained = time.time()
                    print("Time to run one beta-r  training: " + str(time_beta_r_trained - time_beta_r_start)+" seconds")
                    time_beta_r_start_prediction = time.time()
                    PredValidation = DDF.PredictIntoTheFuture(F,PreLength,Current_test[bias-1:],Pdata[bias-1-(D-1)*tau:])
                    time_beta_r_end_prediction =  time.time()
                    # plt.figure()

                    print("Time to run one beta-r  prediction: " + str(time_beta_r_end_prediction - time_beta_r_start_prediction)+" seconds")

                    # Tau8
                    plt.figure(figsize=(20,10))
                    plt.plot(X,Pdata[bias:bias + PreLength],label = 'True Voltage', color = 'black')
                    plt.plot(X,PredValidation[tau*(D-1)+1:tau*(D-1)+PreLength+1],'r--',label = 'Prediction')
                    plt.xlabel('Time ('+str(Time_units)+')',fontsize=20)
                    plt.ylabel('Voltage ('+str(Voltage_units)+')',fontsize=20)
                    plt.legend()
                    plt.title(title,fontsize=20)
                    # plt.show()
                    plt.savefig(directory_to_store_plots+title+'.png')
                    # plt.show
                    print(f"Done with train {directory_to_read_input_data_test} and test{directory_to_read_input_data_test} for (tau,D,beta,R) {(tau,D,beta,R)}")
                    time_beta_r_end = time.time()
                    print("Time to run one beta-r train plus prediction: " + str(time_beta_r_end - time_beta_r_start)+" seconds")

                    print("Saving training and testing data to files")
                    training_times =   loaded_t_train[:train_timestep_end - 1000 + tau*D] # most time-lagged variable V(t-tau*D) goes from data[0:length]. V(t) goes from data[tau*D:length+tau*D]
                    used_Voltage_train = loaded_V_train[:train_timestep_end - 1000 + tau*D]
                    used_Current_train = loaded_I_train[:train_timestep_end - 1000 + tau*D]
                    testing_times = (loaded_t_test[test_timestep_start:total_num_timesteps_in_data_test])[bias:bias + PreLength]
                    used_Voltage_test =  (loaded_V_test[test_timestep_start:total_num_timesteps_in_data_test])[bias:bias + PreLength]
                    used_Current_test =  (loaded_I_test[test_timestep_start:total_num_timesteps_in_data_test])[bias:bias + PreLength]
                    Voltage_pred =  PredValidation[tau*(D-1)+1:tau*(D-1)+PreLength+1]

                    # plt.figure()
                    # plt.plot(Voltage_pred)
                    # plt.plot(used_Current_test)
                    # plt.plot(Voltage_pred)
                    # plt.title("Testing Stimulus and Predicted Voltage")
                    # plt.xlabel("Time")
                    # plt.ylabel("Voltage or Current")
                    # plt.show()
                    #
                    # plt.figure()
                    # plt.plot(Voltage_pred)
                    # plt.plot((loaded_I_train[test_timestep_start:total_num_timesteps_in_data_test])[bias:bias + PreLength])
                    # plt.plot(Voltage_pred)
                    # plt.title("Source Stimulus for training and Predicted Voltage")
                    # plt.xlabel("Time")
                    # plt.ylabel("Voltage or Current")
                    # plt.show()


                    # In[8]:
                    print("Saving data")
                    # Prediction and Truth Plotting
                    save_utilities.save_text(data=Pdata[bias:bias + PreLength],
                                             a_str=save_and_or_display,
                                             save_location=directory_to_store_txt_data + "prediction_and_truth/" + title + "_voltage_truth.txt")
                    save_utilities.save_text(data=Voltage_pred,
                                             a_str=save_and_or_display,
                                             save_location=directory_to_store_txt_data + "prediction_and_truth/" + title + "_voltage_prediction.txt")
                    save_utilities.save_text(data=X,
                                             a_str=save_and_or_display,
                                             save_location=directory_to_store_txt_data + "prediction_and_truth/" + title + "_time.txt")




                    # Coefficient Plotting
                    # plotting_utilities.plotting_quantity(x_arr=range(len(np.sort(DDF.W))), y_arr=np.sort(DDF.W), title="RBF Coefficients (Sorted)",
                    #                                      xlabel='Index (Sorted)',
                    #                                      ylabel="RBF Coefficient Value",
                    #                                      save_and_or_display=save_and_or_display,
                    #                                      save_location=directory_to_store_plots +"weights/"+ title+"_RBF_Coefficients_(Sorted).png")
                    # save_utilities.save_text(data=np.sort(DDF.W),
                    #                          a_str=save_and_or_display,
                    #                          save_location=directory_to_store_txt_data +"weights/"+ title + "_RBF_Coefficients_(Sorted).txt")


                    # plotting_utilities.plotting_quantity(x_arr=range(len(DDF.W)), y_arr=DDF.W, title="RBF Coefficients (Unsorted)",
                    #                                      xlabel='Index (Unsorted)',
                    #                                      ylabel="RBF Coefficient Value",
                    #                                      save_and_or_display=save_and_or_display,
                    #                                      save_location=directory_to_store_plots +"weights/"+ title+"_RBF_Coefficients_(Unsorted).png")
                    # save_utilities.save_text(data=DDF.W,
                    #                          a_str=save_and_or_display,
                    #                          save_location=directory_to_store_txt_data +"weights/"+ title + "_RBF_Coefficients_(Unsorted).txt")

                    # plotting_utilities.plotting_quantity(x_arr=range(len(np.sort(Center[:, 0]))), y_arr=np.sort(Center[:, 0]), title="Centers",
                    #                                      xlabel="Sorted centers index",
                    #                                      ylabel="Voltage ("+str(Voltage_units)+")",
                    #                                      save_and_or_display=save_and_or_display,
                    #                                      save_location=directory_to_store_plots +"centers/"+ title+"_Sorted_centers_vs_index.png")
                    # save_utilities.save_text(data=np.sort(Center[:, 0]),
                    #                          a_str=save_and_or_display,
                    #                          save_location=directory_to_store_txt_data +"centers/"+ title + "_Sorted_centers_vs_index.txt")


                    plt.figure()
                    plt.scatter(Center[:, 0], DDF.W[:-1])
                    plt.title("Weight as a function of center voltage")
                    plt.xlabel("Center voltage ("+str(Voltage_units)+")")
                    plt.ylabel("Weight (Coeff of RBF)")
                    if not os.path.isdir(directory_to_store_plots +"weights/"):
                        os.makedirs(directory_to_store_plots +"weights/")
                    plt.savefig(directory_to_store_plots +"weights/"+ title+ "Weight(center_voltage).png")
                    if "display" in save_and_or_display:
                        plt.show()
                    if "display" not in save_and_or_display:
                        plt.close("all")
                    # save_utilities.save_text(data=np.column_stack((Center[:, 0], DDF.W[:-1])),
                    #                          a_str=save_and_or_display,
                    #                          save_location=directory_to_store_txt_data +"weights/"+ title + "_Weight(center_voltage).txt")
                    #
                    # ========= Dawei Li's Convolution Spiking MSE Code ===============
                    # sampling frequency: 50kHz, step size: 0.02ms
                    # range of sigma_G
                    # load file here instead of regenerating by commenting out above and uncommenting next four lines
                    # NoCenters = 500
                    # title = str(neuron_name)+'_'+str(a_filename[:-4])+' with tstep='+str(TT)+' '+str(Time_units)+', D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train TSteps = '+str(length)+', Centers = '+str(NoCenters)+', tau = '+str(tau)
                    # X = np.loadtxt(directory_to_store_txt_data + "prediction_and_truth/"+trial_number_string+"/" + title + "_time.txt")
                    # Voltage_pred = np.loadtxt(directory_to_store_txt_data + "prediction_and_truth/"+trial_number_string+"/" + title + "_voltage_prediction.txt")
                    # used_Voltage_test = np.loadtxt(directory_to_store_txt_data + "prediction_and_truth/"+trial_number_string+"/" + title + "_voltage_truth.txt")
                    # print("Testing for "+str(title)+";"+" Trial "+str(trial_index))

                    cost_list = []
                    for sigma_c_index, sigma_convolution in enumerate(sigma_c_list):
                        # sigma_convolution = 200
                        if sigma_convolution == 0:
                            predict_conv = Voltage_pred
                            true_conv = used_Voltage_test
                        else:
                            predict_conv = gaussian_filter1d(Voltage_pred, sigma_convolution, truncate=4)
                            true_conv = gaussian_filter1d(used_Voltage_test, sigma_convolution, truncate=4)
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.plot(X, used_Voltage_test, 'g', label='True Voltage')
                        ax.plot(X, true_conv, label='True Voltage (convolved)', color='black')
                        ax.plot(X, predict_conv, 'r--', label='Prediction (convolved)')
                        ax.set_title("Convolved V(t) for Sigma="+str(sigma_convolution))
                        ax.set_xlabel("Time ("+str(Time_units)+")")
                        ax.set_ylabel("Convolved Voltage ("+str(Voltage_units)+")")
                        # if "current" in title.lower():
                        #     ax.get_lines()[0].set_color("orange")
                        # if "voltage" in title.lower():
                        #     ax.get_lines()[0].set_color("blue")
                        # plt.xlim((12.5,13.5))
                        # save_utilities.save_and_or_display_plot(fig, "save", directory_to_store_plots + "convolution_MSE_metric/"+trial_number_string+"/" + title + "_Convolved_waveforms_sigma="+str(sigma_convolution)+".png")
                        plt.close(fig)
                        cost = (np.sum((predict_conv - true_conv) ** 2)) / len(true_conv)
                        # cost_list.append(cost) # m
                        if varying_hyperparam.lower() == "r":
                            conv_MSE_trials_array[R_index,trial_index,sigma_c_index] = cost
                        if varying_hyperparam.lower() == "d":
                            conv_MSE_trials_array[D_index,trial_index,sigma_c_index] = cost
                        if varying_hyperparam.lower() == "tau":
                            conv_MSE_trials_array[tau_index,trial_index,sigma_c_index] = cost


                    print("COST WAS: "+str(cost))

                    # convert the unit of sigma_G from step to ms
                    sigma_range_in_ms = np.array(sigma_c_list) * TT

                    fig2 = plt.figure()
                    ax = fig2.add_subplot(111)
                    # ax.plot(sigma_range_in_ms, conv_MSE_trials_array[R_index, trial_index], linewidth=2)
                    if varying_hyperparam.lower() == "r":
                        ax.plot(sigma_range_in_ms, conv_MSE_trials_array[R_index, trial_index], linewidth=2)
                    if varying_hyperparam.lower() == "d":
                        ax.plot(sigma_range_in_ms, conv_MSE_trials_array[D_index, trial_index], linewidth=2)
                    if varying_hyperparam.lower() == "tau":
                        ax.plot(sigma_range_in_ms, conv_MSE_trials_array[tau_index, trial_index], linewidth=2)

                    ax.set_title("Convolved V(t)")
                    ax.set_xlabel(r"$\sigma_C$("+str(Time_units)+")") # Check to make sure this is accurate
                    ax.set_ylabel(r"cost value(mV$^2$)")
                    if "current" in title.lower():
                        ax.get_lines()[0].set_color("orange")
                    if "voltage" in title.lower():
                        ax.get_lines()[0].set_color("blue")
                    # save_utilities.save_and_or_display_plot(fig2, "save", directory_to_store_plots +"convolution_MSE_metric/"+trial_number_string+"/" +title+"_MSE_metric_vs_sigma.png")
                    # plt.close(fig2)


print("Making convolutional MSE plots")

# np.save(directory_to_store_txt_data+"conv_MSE_"+str(varying_hyperparam)+"_trials_array.npy", conv_MSE_trials_array)
conv_MSE_trials_array = np.load(directory_to_store_txt_data+"conv_MSE_"+str(varying_hyperparam)+"_trials_array.npy")
NoCenters = 500

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
# "User Side Code Separate Train and Test Same Epoch Different CM Auto File Loop"
# on June 6, 2022. I'm using new function definitions to make this file a cleaned up version of that file.

# In[2]:
# modify this
save_and_or_display = "save"

# epoch = None # also called "episode". set to None if not specified
# tau_arr = np.array([7,10])#np.array(range(10, 20)) # math notation: range(2,10) = all integers in bounds [2,9)
# D_arr = np.array([2,3])#np.array(range(2, 10)) # math notation: range(2,10) = all integers in bounds [2,9)
# beta_arr = np.array(np.power(10.0,[-1,0,1]))#np.array(np.power(10.0,range(-3,3))) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
# R_arr = np.array(np.power(10.0,[-1,0,1])) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
# file_extension = "abf" # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase

# 920061fe 2014_12_11_0017 (Fully run already)
# tau_arr = np.array([2,3,4])#np.array(range(10, 20)) # math notation: range(2,10) = all integers in bounds [2,9)
# D_arr = np.array([4,5,6])#np.array(range(2, 10)) # math notation: range(2,10) = all integers in bounds [2,9)
# beta_arr = np.array(np.power(10.0,[-2,-1,0]))#np.array(np.power(10.0,range(-3,3))) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
# R_arr = np.array(np.power(10.0,[-2,-1,0])) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
# file_extension = "abf" # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase

# 32425a75 2014_09_10_0013_VIt (Fully run already)
# tau_arr = np.array([2,3,4])#np.array(range(10, 20)) # math notation: range(2,10) = all integers in bounds [2,9)
# D_arr = np.array([6,7,8])#np.array(range(2, 10)) # math notation: range(2,10) = all integers in bounds [2,9)
# beta_arr = np.array(np.power(10.0,[0,1,2]))#np.array(np.power(10.0,range(-3,3))) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
# R_arr = np.array(np.power(10.0,[-2,-1,0])) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
# file_extension = "abf" # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase

# biocm_simulations (In process)
# tau_arr = np.array([2,3,4])#np.array(range(10, 20)) # math notation: range(2,10) = all integers in bounds [2,9)
# D_arr = np.array([2,4,6])#np.array(range(2, 10)) # math notation: range(2,10) = all integers in bounds [2,9)
# beta_arr = np.array(np.power(10.0,[-1,0,1]))#np.array(np.power(10.0,range(-3,3))) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
# R_arr = np.array(np.power(10.0,[-1,0,1])) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
# file_extension = "mat" # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase

# Testing MSE convolutional sigma with hyperparams that worked for Colpitts-driven Red 171 Neuron 2 (Epoch 1).
tau_arr = np.array([10])#np.array(range(10, 20)) # math notation: range(2,10) = all integers in bounds [2,9)
D_arr = np.array([10])#np.array(range(2, 10)) # math notation: range(2,10) = all integers in bounds [2,9)
beta_arr = np.array(np.power(10.0,[-3]))#np.array(np.power(10.0,range(-3,3))) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
R_arr = np.array(np.power(10.0,[-5,-4,-3,-2,-1])) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
file_extension = "txt" # string; examples: "atf" or "txt" (case sensitive); don't include period; lowercase

# specify what the neuron names are in the file titles here:
neuron_name_list = ['32425a75',
                     '920061fe'] # example: ['32425a75', '920061fe'] are two CM neurons from Meliza's 2014 data
Current_units = "pA"
Voltage_units = "mV"
Time_units = "ms"
TT = 0.02 # delta t in Time_units units, time between samples if not specified through loaded files

# Data directory to recursively load data from:
root_directory = "Data2022-50KhZ/7-7-2022/Red 171/Neuron 2/"#"biocm_simulations/"#"cm_ddf/data/"#"Data2022-50KhZ/" # example: "HVC_biocm_data/simulations/" ; Include the final "/"

# Use only this file:
files_to_evaluate = ["epoch_1.txt"]#"biocm_phasic_lzo_1_1_10_100_200.mat"] # leave this list empty if you want to evaluate all files in root_directory recursively

do_not_use_list = ["2014_09_10_0001.abf",
                   "2014_09_10_0002.abf",
                   "2014_09_10_0003.abf"# file attached to Meliza's data when I received it said not to use these. The input signals are not suitable for training
                   ] # bad data for RBF training

FPS_xlim= (0,0.175)

fraction_of_data_for_training = 4.0/6.0

sigma_c_list = np.array([5, 20, 50, 100, 200, 400]) # list of convolutional gaussian sigma for MSE calc

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
    # if "epoch_5" not in a_path:
    #     continue
    # if "Neuron 2" not in a_path:
    #     continue
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
            if 'Current' in a_path:
                voltage_filepath = a_path.replace('Current','Voltage')
            loaded_V = np.loadtxt(voltage_filepath)
            loaded_I = np.loadtxt(a_path)
        loaded_t = TT*np.array(range(len(loaded_V)))

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
    #--------------- Current FPS Plots --------------
    # Training Current with no modifications
    fig = plotting_utilities.plotting_quantity(x_arr = freq_array, y_arr = power_spectrum/np.max(np.abs(power_spectrum)),
                                               title = "Power(freq) of current_train (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")",
                                               xlabel = "Frequency (kHz)",
                                               ylabel = "Normalized Power (1.0 = max from whole spectrum)",
                                               save_and_or_display= save_and_or_display,
                                               save_location=directory_to_store_plots+"Fourier_analysis/"+"Power spectrum of training current (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")_full_data.png",
                                               xlim=None)
    save_utilities.save_text(data = np.column_stack((freq_array, power_spectrum)),
                             a_str = save_and_or_display,
                             save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_current-full_data.txt")


    # Training Current - No index 0, Normalized, many windows
    for window_size in [0.175]:#5, 75, 150, 300]:
        final_index = int(round(float(window_size) / delta_freq))
        fig = plotting_utilities.plotting_quantity(x_arr = freq_without_0_index, y_arr = normalized_power_spec_without_0_index,
                                                   title = "Power(freq) of current_train (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")",
                                                   xlabel = "Frequency (kHz)",
                                                   ylabel = "Normalized Power (1.0 = max from [1:])",
                                                   save_and_or_display= save_and_or_display,
                                                   save_location=directory_to_store_plots+"Fourier_analysis/"+"Power spectrum of training current (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")_last_freq_shown="+str(window_size)+".png",
                                                   xlim=(0, window_size))
        save_utilities.save_text(data = np.column_stack((freq_without_0_index[:final_index], normalized_power_spec_without_0_index[:final_index])),
                                 a_str = save_and_or_display,
                                 save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_current-_last_freq_shown="+str(window_size)+".txt")


    # --------------- Voltage FPS Plots --------------
    power_spectrum = FPS_list[1]
    # frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
    normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))


    # Training Voltage with no modifications
    fig = plotting_utilities.plotting_quantity(x_arr = freq_array, y_arr = power_spectrum/np.max(np.abs(power_spectrum)),
                                               title = "Power(freq) of voltage_train (Neuron " + str(neuron_name) + '_' + str(a_filename[:-4]) + ")",
                                               xlabel = "Frequency (kHz)",
                                               ylabel = "Normalized Power (1.0 = max from whole spectrum)",
                                               save_and_or_display= save_and_or_display,
                                               save_location=directory_to_store_plots +"Fourier_analysis/"+ "Power spectrum of training voltage (Neuron " + str(neuron_name) + '_' + str(a_filename[:-4]) + ")_full_data.png",
                                               xlim=None)
    save_utilities.save_text(data = np.column_stack((freq_array, power_spectrum)),
                             a_str = save_and_or_display,
                             save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_voltage-full_data.txt")


    # Training Current - No index 0, Normalized, many windows
    for window_size in [0.175]:#5, 75, 150, 300]:
        final_index = int(round(float(window_size) / delta_freq))
        fig = plotting_utilities.plotting_quantity(x_arr = freq_without_0_index, y_arr = normalized_power_spec_without_0_index,
                                                   title = "Power(freq) of voltage_train (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")",
                                                   xlabel = "Frequency (kHz)",
                                                   ylabel = "Normalized Power (1.0 = max from [1:])",
                                                   save_and_or_display= save_and_or_display,
                                                   save_location=directory_to_store_plots+"Fourier_analysis/"+"Power spectrum of training voltage (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")_last_freq_shown="+str(window_size)+".png",
                                                   xlim=(0, window_size))
        save_utilities.save_text(data = np.column_stack((freq_without_0_index[:final_index], normalized_power_spec_without_0_index[:final_index])),
                                 a_str = save_and_or_display,
                                 save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_voltage-_last_freq_shown="+str(window_size)+".txt")

    # ===============================  END OF POWER SPECTRA  =====================================
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


    # In[6]:
    num_R_trials = 6
    conv_MSE_R_trials_array = np.zeros((np.shape(R_arr)[0], num_R_trials, sigma_c_list.shape[0]))  # scalar MSE for each R (row) and trial (column)
    print("Shape of conv_MSE_R_trials_array: " +str(np.shape(conv_MSE_R_trials_array)))
    # =============================== Training and Prediction  =====================================
    print("Beginning Training and prediction for "+str(a_path))
    for R_trial_index in range(num_R_trials):
        for tau in tau_arr:
            for D in D_arr:
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
                temp_array = copy.deepcopy(Xdata)
                temp_array[temp_array<-50]=-100
                # Centers_above_thresh = DDF.KmeanCenter(temp_array,NoCenters_above_thresh,D,length,tau);
                # Center = np.concatenate((Centers_k_means,Centers_above_thresh),axis=0)
                Center = Centers_k_means

                NoCenters = np.shape(Center)[0]
                print(NoCenters)
                print("Centers:"+str(Center.shape))
                np.savetxt('centers/Center '+str(neuron_name)+'_'+str(a_filename[:-4])+'(D,tau,NumCenters)='+str((D,tau,NoCenters))+'.txt',Center)
                Center = np.loadtxt('centers/Center '+str(neuron_name)+'_'+str(a_filename[:-4])+'(D,tau,NumCenters)='+str((D,tau,NoCenters))+'.txt')
                print("Centers used: \n "+str(Center[:6,0]))


                stim_train = Current_train
                Pdata = Voltage_test
                bias = tau*(D-1)+1#50 # should be larger than tau*(D-1) or something like that
                # X = np.arange(bias,bias+PreLength*TT,TT)
                X = Time_test[bias:bias+PreLength]#bias -1 maybe?

                time_preparing_to_run_beta_r = time.time()
                print("Time to reach right before beta_r loop: "+str(time_preparing_to_run_beta_r-time_start))
                # In[7]:
                for beta in beta_arr:
                    for R_index, R in enumerate(R_arr):
                        # if (not math.isclose(beta,1.0)) or (not math.isclose(R,0.01)):
                        #     continue
                        print("(tau, D, beta, R) = " + str((tau, D, beta,R)))
                        time_beta_r_start = time.time()
                        title = str(neuron_name)+'_'+str(a_filename[:-4])+' with tstep='+str(TT)+' '+str(Time_units)+', D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train TSteps = '+str(length)+', Centers = '+str(NoCenters)+', tau = '+str(tau)
                        # print(R)
                        # print("Shape of Xdata is now "+str(Xdata.shape))

                        F = DDF.FuncApproxF(Xdata,length,Center,beta,R,D,stim_train,tau)
                        time_beta_r_trained = time.time()
                        print("Time to run one beta-r  training: " + str(time_beta_r_trained - time_beta_r_start))
                        time_beta_r_start_prediction = time.time()
                        PredValidation = DDF.PredictIntoTheFuture(F,PreLength,Current_test[bias-1:],Pdata[bias-1-(D-1)*tau:])
                        time_beta_r_end_prediction =  time.time()
                        print("Time to run one beta-r  prediction: " + str(time_beta_r_end_prediction - time_beta_r_start_prediction))

                        # Tau8
                        plt.figure(figsize=(20,10))
                        plt.plot(X,Pdata[bias:bias + PreLength],label = 'True Voltage', color = 'black')
                        plt.plot(X,PredValidation[tau*(D-1)+1:tau*(D-1)+PreLength+1],'r--',label = 'Prediction')
                        plt.xlabel('Time ('+str(Time_units)+')',fontsize=20)
                        plt.ylabel('Voltage ('+str(Voltage_units)+')',fontsize=20)
                        plt.legend()
                        plt.title(title,fontsize=20)
                        #plt.savefig('Validation Prediction Result')
                        plt.savefig(directory_to_store_plots+title+'.png')
                        # plt.show
                        print("Done with "+str((directory_to_read_input_data)+str(neuron_name)+str((tau,D,beta,R))))
                        time_beta_r_end = time.time()
                        print("Time to run one beta-r train plus prediction: " + str(time_beta_r_end - time_beta_r_start))

                        print("Saving training and testing data to files")
                        training_times =   loaded_t[:train_timestep_end - 1000 + tau*D] # most time-lagged variable V(t-tau*D) goes from data[0:length]. V(t) goes from data[tau*D:length+tau*D]
                        used_Voltage_train = loaded_V[:train_timestep_end - 1000 + tau*D]
                        used_Current_train = loaded_I[:train_timestep_end - 1000 + tau*D]
                        testing_times = (loaded_t[train_timestep_end:total_num_timesteps_in_data])[bias:bias + PreLength]
                        used_Voltage_test =  (loaded_V[train_timestep_end:total_num_timesteps_in_data])[bias:bias + PreLength]
                        used_Current_test =  (loaded_I[train_timestep_end:total_num_timesteps_in_data])[bias:bias + PreLength]
                        Voltage_pred =  PredValidation[tau*(D-1)+1:tau*(D-1)+PreLength+1]

                        # In[8]:

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
                        plotting_utilities.plotting_quantity(x_arr=range(len(np.sort(DDF.W))), y_arr=np.sort(DDF.W), title="RBF Coefficients (Sorted)",
                                                             xlabel='Index (Sorted)',
                                                             ylabel="RBF Coefficient Value",
                                                             save_and_or_display=save_and_or_display,
                                                             save_location=directory_to_store_plots +"weights/"+ title+"_RBF_Coefficients_(Sorted).png")
                        save_utilities.save_text(data=np.sort(DDF.W),
                                                 a_str=save_and_or_display,
                                                 save_location=directory_to_store_txt_data +"weights/"+ title + "_RBF_Coefficients_(Sorted).txt")


                        plotting_utilities.plotting_quantity(x_arr=range(len(DDF.W)), y_arr=DDF.W, title="RBF Coefficients (Unsorted)",
                                                             xlabel='Index (Unsorted)',
                                                             ylabel="RBF Coefficient Value",
                                                             save_and_or_display=save_and_or_display,
                                                             save_location=directory_to_store_plots +"weights/"+ title+"_RBF_Coefficients_(Unsorted).png")
                        save_utilities.save_text(data=DDF.W,
                                                 a_str=save_and_or_display,
                                                 save_location=directory_to_store_txt_data +"weights/"+ title + "_RBF_Coefficients_(Unsorted).txt")

                        plotting_utilities.plotting_quantity(x_arr=range(len(np.sort(Center[:, 0]))), y_arr=np.sort(Center[:, 0]), title="Centers",
                                                             xlabel="Sorted centers index",
                                                             ylabel="Voltage ("+str(Voltage_units)+")",
                                                             save_and_or_display=save_and_or_display,
                                                             save_location=directory_to_store_plots +"centers/"+ title+"_Sorted_centers_vs_index.png")
                        save_utilities.save_text(data=np.sort(Center[:, 0]),
                                                 a_str=save_and_or_display,
                                                 save_location=directory_to_store_txt_data +"centers/"+ title + "_Sorted_centers_vs_index.txt")


                        plt.figure()
                        plt.scatter(Center[:, 0], DDF.W[:-1])
                        plt.title("Weight as a function of center voltage")
                        plt.xlabel("Center voltage ("+str(Voltage_units)+")")
                        plt.ylabel("Weight (Coeff of RBF)")
                        plt.savefig(directory_to_store_plots +"weights/"+ title+ "Weight(center_voltage).png")
                        if "display" in save_and_or_display:
                            plt.show()
                        if "display" not in save_and_or_display:
                            plt.close("all")
                        save_utilities.save_text(data=np.column_stack((Center[:, 0], DDF.W[:-1])),
                                                 a_str=save_and_or_display,
                                                 save_location=directory_to_store_txt_data +"weights/"+ title + "_Weight(center_voltage).txt")

                        # ========= Dawei Li's Convolution Spiking MSE Code ===============
                        # simpling frequency: 50kHz, step size: 0.02ms
                        # range of sigma_G
                        # load file here instead of regenerating by commenting out above and uncommenting next four lines
                        # X = np.loadtxt(directory_to_store_txt_data + "prediction_and_truth/" + "_epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_time.txt")
                        # Voltage_pred = np.loadtxt(directory_to_store_txt_data + "prediction_and_truth/" + "_epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_voltage_prediction.txt")
                        # used_Voltage_test = np.loadtxt(directory_to_store_txt_data + "prediction_and_truth/" + "_epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_voltage_truth.txt")
                        # title = '_epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_time.txt'


                        cost_list = []
                        for sigma_c_index, sigma_convolution in enumerate(sigma_c_list):
                            # sigma_convolution = 200
                            predict_conv = gaussian_filter1d(Voltage_pred, sigma_convolution, truncate=2)
                            true_conv = gaussian_filter1d(used_Voltage_test, sigma_convolution, truncate=2)
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
                            save_utilities.save_and_or_display_plot(fig, "save", directory_to_store_plots + "convolution_MSE_metric/" + title + "_Convolved_waveforms_sigma="+str(sigma_convolution)+".png")
                            plt.close(fig)
                            cost = (np.sum((predict_conv - true_conv) ** 2)) / len(true_conv)
                            # cost_list.append(cost) # m
                            conv_MSE_R_trials_array[R_index,R_trial_index,sigma_c_index] = cost
                        print("COST WAS: "+str(cost))

                        # convert the unit of sigma_G from step to ms
                        sigma_range_in_ms = np.array(sigma_c_list) * TT

                        fig2 = plt.figure()
                        ax = fig2.add_subplot(111)
                        ax.plot(sigma_range_in_ms, conv_MSE_R_trials_array[R_index, R_trial_index], linewidth=2)
                        ax.set_title("Convolved V(t)")
                        ax.set_xlabel(r"$\sigma_C$("+str(Time_units)+")")
                        ax.set_ylabel(r"cost value(mV$^2$)")
                        if "current" in title.lower():
                            ax.get_lines()[0].set_color("orange")
                        if "voltage" in title.lower():
                            ax.get_lines()[0].set_color("blue")
                        save_utilities.save_and_or_display_plot(fig2, "save", directory_to_store_plots +"convolution_MSE_metric/"+ title+"_MSE_metric_vs_sigma.png")
                        plt.close(fig2)

print("Making convolutional MSE plots")
for sigma_c_index, sigma_c in enumerate(sigma_c_list):
    MSE_range_fig = plt.figure()
    ax = MSE_range_fig.add_subplot(111)
    y_err1 = np.std(conv_MSE_R_trials_array[:,:,sigma_c_index], axis=1)
    ax.errorbar(np.log10(R_arr), np.average(conv_MSE_R_trials_array[:,:,sigma_c_index],axis=1), yerr=y_err1)
    for i in range(num_R_trials):
        ax.scatter(np.log10(R_arr), conv_MSE_R_trials_array[:,i,sigma_c_index])
    ax.set_xlabel("log10(R)")
    ax.set_ylabel("MSE_after_convolution")
    title = str(neuron_name) + '_' + str(a_filename[:-4]) + ' with tstep=' + str(TT) + ' ' + str(
        Time_units) + ', D = ' + str(D) + ', Beta = ' + str("{:.1e}".format(beta)) + ', R = variable' + ' Train TSteps = ' + str(length) + ', Centers = ' + str(NoCenters) + ', tau = ' + str(tau)
    save_utilities.save_and_or_display_plot(MSE_range_fig, "save", directory_to_store_plots +"convolution_MSE_metric/"+ title+"_MSE_vs_R_for_sigma="+str(sigma_c)+".png")
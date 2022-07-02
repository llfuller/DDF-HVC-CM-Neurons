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

import numpy.fft
import math

#imports for Meliza 2014 CM data:
import os
# end of imports for Meliza 2014 CM data

random.seed(2022)
np.random.seed(2022)

# This code is an edit branching off of Python code
# "User Side Code Separate Train and Test Same Epoch Different CM Auto File Loop"
# on June 6, 2022. I'm using new function definitions to make this file a cleaned up version of that file.

# In[2]:


# modify this
epoch = None # also called "episode". set to None if not specified
tau_arr = np.array(range(2, 10)) # math notation: range(2,10) = all integers in bounds [2,9)
D_arr = np.array(range(2, 10)) # math notation: range(2,10) = all integers in bounds [2,9)
beta_arr = np.array(np.power(10.0,range(-3,3))) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
R_arr = np.array(np.power(10.0,range(-3,3))) #range(-3,3) makes array go from 1e-3 to 1e2, not 1e3
file_extension = "mat" # string; examples: "atf" or "txt" (case sensitive); don't include period

# specify what the neuron names are in the file titles here:
neuron_name_list = ['32425a75',
                     '920061fe'] # example: ['32425a75', '920061fe'] are two CM neurons from Meliza's 2014 data

# # specify which directories you want
# directories_list = ['CM_data/'+str(neuron_name_list[0])+'/',
#                     'CM_data/'+str(neuron_name_list[1]+'/')
#                     ]

# In[3]:
print("1")
# Data directory to recursively load data from:
root_directory = "HVC_biocm_data/simulations/" # example: "HVC_biocm_data/simulations/" ; Include the final "/"
print("23")
# ======== do not modify below ==========
full_paths_list = glob.glob(root_directory+"*."+str(save_utilities.glob_extension_case_string_builder(file_extension)),
                            recursive=True)
print(full_paths_list)
neuron_name = "" # leave as "" if no neuron name is found
# Files to ignore within directory:
do_not_use_list = ["2014_09_10_0001.abf",
                   "2014_09_10_0002.abf",
                   "2014_09_10_0003.abf"# file attached to Meliza's data when I received it said not to use these. The input signals are horrible
                   ] # bad data

# Code for 2014 Meliza CM data
for a_path in full_paths_list:
    print("Checking "+str(a_path))
    last_slash_location = a_path.rfind("/")
    neuron_directory = a_path[:last_slash_location+1] # should include the last slash, but nothing past it
    a_filename = a_path[last_slash_location+1:]
    neuron_name = save_utilities.give_name_if_included_in_path(a_path, neuron_name_list)
    print("================================New File ==================================================")
    if a_filename in do_not_use_list:
        continue # skip this iteration if the filename is on the do_not_use_list
    if a_filename != "2014_09_10_0013.abf": #"2014_12_11_0017.abf":#"2014_09_10_0013.abf":#"2014_12_11_0017.abf":
        continue # want to try predicting only this neuron since it's complex and most interesting
    directory_to_store_plots = "plots/"+neuron_directory+str(a_filename[:-4])+"/"
    # # Code for 2014 Meliza CM data
    current_units = "pA"
    voltage_units = "mV"
    time_units = "ms"

    directory_to_store_txt_data = "data_derived/" + neuron_directory + 'txt_V_I_t/'
    save_utilities.convert_atf_to_VIt_text_file(neuron_directory, a_filename, directory_to_store_txt_data)
    # neuron data/920061fe/2014_12_11_0017.abf from new Meliza data
    imported_data = np.loadtxt(directory_to_store_txt_data+str(a_filename[:-4])+"_VIt.txt")

    loaded_V = imported_data[:,0]
    loaded_I = imported_data[:,1]
    loaded_t = imported_data[:,2]

    total_num_timesteps_in_data = len(loaded_V)
    train_timestep_end = round(total_num_timesteps_in_data*5.0/6.0)
    Voltage_train = loaded_V[:train_timestep_end]
    Current_train = loaded_I[:train_timestep_end]
    Time_train    = loaded_t[:train_timestep_end]
    Voltage_test = loaded_V[train_timestep_end:total_num_timesteps_in_data]
    Current_test = loaded_I[train_timestep_end:total_num_timesteps_in_data]
    Time_test    = loaded_t[train_timestep_end:total_num_timesteps_in_data]
    length = Voltage_train.shape[0]-1000 # - 1000 just to give breathing room
    PreLength = Voltage_test.shape[0]-1000 # - 1000 just to give breathing room
    print("Train timestep end:"+str(train_timestep_end))
    print("Shape of loaded V:"+str(loaded_V.shape))
    print("Shape of Voltage train is "+str(Voltage_train.shape))

    # # make directory to save plots to if it doesn't yet exist
    # if not os.path.isdir(directory_to_store_plots):
    #     os.mkdir(directory_to_store_plots)

    # ===============================  POWER SPECTRA  =====================================
    # FFT Train
    freq_units = "kHz"  # frequency units
    TT = float(loaded_t[2]-loaded_t[1]) # delta t; time between data samples
    sampling_rate = 1.0/TT # frequency (1/ms = kHz)
    FPS_list, freq_array = Fourier_Power_Spectrum.calc_Fourier_power_spectrum([Current_train, Voltage_train], Time_train)
    delta_freq = freq_array[3]-freq_array[2]
    save_and_or_display = "save and display"

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
                                               save_location=directory_to_store_plots+"Power spectrum of training current (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")_full_data.png",
                                               xlim=None)
    save_utilities.save_text(data = np.column_stack((freq_array, power_spectrum)),
                             a_str = save_and_or_display,
                             save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_current-full_data.txt")


    # Training Current - No index 0, Normalized, many windows
    for window_size in [5, 75, 150, 300]:
        final_index = int(round(float(window_size) / delta_freq))
        fig = plotting_utilities.plotting_quantity(x_arr = freq_without_0_index, y_arr = normalized_power_spec_without_0_index,
                                                   title = "Power(freq) of current_train (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")",
                                                   xlabel = "Frequency (kHz)",
                                                   ylabel = "Normalized Power (1.0 = max from [1:])",
                                                   save_and_or_display= save_and_or_display,
                                                   save_location=directory_to_store_plots+"Power spectrum of training current (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")_last_freq_shown="+str(window_size)+".png",
                                                   xlim=(0, window_size))
        save_utilities.save_text(data = np.column_stack((freq_without_0_index[:final_index], normalized_power_spec_without_0_index[:final_index])),
                                 a_str = save_and_or_display,
                                 save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_current-_last_freq_shown="+str(window_size)+".txt")


    # --------------- Voltage FPS Plots --------------
    power_spectrum = FPS_list[1]
    # frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
    normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))


    # Training Voltage with no modifications
    fig = plotting_utilities.plotting_quantity(x_arr = freq_without_0_index, y_arr = power_spectrum/np.max(np.abs(power_spectrum)),
                                               title = "Power(freq) of voltage_train (Neuron " + str(neuron_name) + '_' + str(a_filename[:-4]) + ")",
                                               xlabel = "Frequency (kHz)",
                                               ylabel = "Normalized Power (1.0 = max from whole spectrum)",
                                               save_and_or_display= save_and_or_display,
                                               save_location=directory_to_store_plots + "Power spectrum of training voltage (Neuron " + str(neuron_name) + '_' + str(a_filename[:-4]) + ")_full_data.png",
                                               xlim=None)
    save_utilities.save_text(data = np.column_stack((freq_array, power_spectrum)),
                             a_str = save_and_or_display,
                             save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_voltage-full_data.txt")


    # Training Current - No index 0, Normalized, many windows
    for window_size in [5, 75, 150, 300]:
        final_index = int(round(float(window_size) / delta_freq))
        fig = plotting_utilities.plotting_quantity(x_arr = freq_without_0_index, y_arr = normalized_power_spec_without_0_index,
                                                   title = "Power(freq) of voltage_train (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")",
                                                   xlabel = "Frequency (kHz)",
                                                   ylabel = "Normalized Power (1.0 = max from [1:])",
                                                   save_and_or_display= save_and_or_display,
                                                   save_location=directory_to_store_plots+"Power spectrum of training voltage (Neuron "+str(neuron_name)+'_'+str(a_filename[:-4])+")_last_freq_shown="+str(window_size)+".png",
                                                   xlim=(0, window_size))
        save_utilities.save_text(data = np.column_stack((freq_without_0_index[:final_index], normalized_power_spec_without_0_index[:final_index])),
                                 a_str = save_and_or_display,
                                 save_location = directory_to_store_txt_data + "Fourier_analysis/" + str(a_filename[:-4]) + "_Fourier_Spectrum_training_voltage-_last_freq_shown="+str(window_size)+".txt")

    # ===============================  END OF POWER SPECTRA  =====================================
    # ===============================  Plotting training and testing current and voltage  =====================================
    plotting_utilities.plotting_quantity(x_arr = Time_train, y_arr = Current_train, title = "Training Current",
                                         xlabel = "Time ("+str(time_units)+")", ylabel = "Current ("+str(current_units)+")",
                                         save_and_or_display="save and display",
                                         save_location=directory_to_store_plots+"Train 1 first half Test 1 second half"+" Training Current.png")
    plotting_utilities.plotting_quantity(x_arr = Time_train, y_arr = Voltage_train, title = "Training Voltage",
                                         xlabel = "Time ("+str(time_units)+")", ylabel = "Voltage ("+str(voltage_units)+")",
                                         save_and_or_display="save and display",
                                         save_location=directory_to_store_plots+"Train 1 first half Test 1 second half"+" Training Voltage.png")

    plotting_utilities.plotting_quantity(x_arr = Time_test, y_arr = Current_test, title = "Test Current",
                                         xlabel = "Time ("+str(time_units)+")", ylabel = "Current ("+str(current_units)+")",
                                         save_and_or_display="save and display",
                                         save_location=directory_to_store_plots+"Train 1 first half Test 1 second half"+" Test Current.png")
    plotting_utilities.plotting_quantity(x_arr = Time_test, y_arr = Voltage_test, title = "Test Voltage",
                                         xlabel = "Time ("+str(time_units)+")", ylabel = "Voltage ("+str(voltage_units)+")",
                                         save_and_or_display="save and display",
                                         save_location=directory_to_store_plots+"Train 1 first half Test 1 second half"+" Test Voltage.png")

    # =============================== Training and Prediction  =====================================
    for tau in tau_arr:
        for D in D_arr:
            print("========================New tau and D combination ==================")
            # if tau != tau_specified or D!= D_specified:
            #     continue  # want to try predicting only this neuron since it's complex and most interesting
            time_start = time.time()
            Xdata = Voltage_train
            print("Shape of Xdata is "+str(Xdata.shape))
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

            print("C_km:"+str(Centers_k_means.shape))

            print("temp_arry:"+str(temp_array))
            # print("C_at:"+str(Centers_above_thresh.shape))

            NoCenters = np.shape(Center)[0]
            print(NoCenters)
            print("Centers:"+str(Center.shape))
            np.savetxt('centers/Center '+str(neuron_name)+'_'+str(a_filename[:-4])+'(D,NumCenters)='+str((D,NoCenters))+'.txt',Center)
            Center = np.loadtxt('centers/Center '+str(neuron_name)+'_'+str(a_filename[:-4])+'(D,NumCenters)='+str((D,NoCenters))+'.txt')



            stim_train = Current_train
            Pdata = Voltage_test
            bias = 50 # should be larger than tau*(D-1) or something like that
            X = np.arange(bias,bias+PreLength*TT,TT)
            time_preparing_to_run_beta_r = time.time()
            print("Time to reach right before beta_r loop: "+str(time_preparing_to_run_beta_r-time_start))

            for beta in beta_arr:
                for R in R_arr:
                    # if (not math.isclose(beta,1.0)) or (not math.isclose(R,0.01)):
                    #     continue
                    print("(beta, R) = " + str((beta,R)))
                    time_beta_r_start = time.time()
                    title = str(neuron_name)+'_'+str(a_filename[:-4])+' with '+str(TT)+' time step, D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train Time = '+str(length)+', Centers = '+str(NoCenters)+', tau = '+str(tau)
                    print(R)
                    print("Shape of Xdata is now "+str(Xdata.shape))
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
                    plt.xlabel('time (ms)',fontsize=20)
                    plt.ylabel('Voltage',fontsize=20)
                    plt.legend()
                    plt.title(title,fontsize=20)
                    #plt.savefig('Validation Prediction Result')
                    plt.savefig(directory_to_store_plots+title+'.png')
                    # plt.show
                    print("Done with "+str((neuron_directory)+str(neuron_name)+str((tau,D,beta,R))))
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


                    plt.figure()
                    plt.plot(training_times, used_Voltage_train)
                    plt.title("Training Voltage")
                    plt.plot()

                    plt.figure()
                    plt.plot(training_times, used_Current_train)
                    plt.title("Training Current")
                    plt.plot()

                    plt.figure()
                    plt.plot(testing_times, used_Voltage_test)
                    plt.title("Testing Voltage")
                    plt.plot()

                    plt.figure()
                    plt.plot(testing_times, used_Current_test)
                    plt.title("Testing Current")
                    plt.plot()

                    plt.show()


                    # Coefficient Plotting
                    plotting_utilities.plotting_quantity(x_arr=range(len(np.sort(DDF.W))), y_arr=np.sort(DDF.W), title="RBF Coefficients (Sorted)",
                                                         xlabel='Index (Sorted)',
                                                         ylabel="RBF Coefficient Value",
                                                         save_and_or_display="save and display",
                                                         save_location=directory_to_store_plots + title+"_RBF_Coefficients_(Sorted).png")
                    save_utilities.save_text(data=np.sort(DDF.W),
                                             a_str=save_and_or_display,
                                             save_location=directory_to_store_txt_data + title + "_RBF_Coefficients_(Sorted).txt")


                    plotting_utilities.plotting_quantity(x_arr=range(len(DDF.W)), y_arr=DDF.W, title="RBF Coefficients (Unsorted)",
                                                         xlabel='Index (Unsorted)',
                                                         ylabel="RBF Coefficient Value",
                                                         save_and_or_display="save and display",
                                                         save_location=directory_to_store_plots + title+"_RBF_Coefficients_(Unsorted).png")
                    save_utilities.save_text(data=DDF.W,
                                             a_str=save_and_or_display,
                                             save_location=directory_to_store_txt_data + title + "_RBF_Coefficients_(Unsorted).txt")

                    plotting_utilities.plotting_quantity(x_arr=range(len(np.sort(Center[:, 0]))), y_arr=np.sort(Center[:, 0]), title="Centers",
                                                         xlabel="Sorted centers index",
                                                         ylabel="Voltage",
                                                         save_and_or_display="save and display",
                                                         save_location=directory_to_store_plots + title+"_Sorted_centers_vs_index.png")
                    save_utilities.save_text(data=np.sort(Center[:, 0]),
                                             a_str=save_and_or_display,
                                             save_location=directory_to_store_txt_data + title + "_Sorted_centers_vs_index.txt")


                    plt.figure()
                    plt.scatter(Center[:, 0], DDF.W[:-1])
                    plt.title("Weight as a function of center voltage (unsure)")
                    plt.xlabel("Center voltage")
                    plt.ylabel("Weight (Coeff of RBF)")
                    plt.savefig(directory_to_store_plots + title+ "Weight(center_voltage).png")
                    plt.show()
                    save_utilities.save_text(data=np.column_stack((Center[:, 0], DDF.W[:-1])),
                                             a_str=save_and_or_display,
                                             save_location=directory_to_store_txt_data + title + "_Weight(center_voltage).txt")

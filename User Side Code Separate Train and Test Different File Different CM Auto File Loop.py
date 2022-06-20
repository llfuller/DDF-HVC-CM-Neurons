#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TimeDelay_Neuron_DDF_GaussianForm import *
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import copy
import numpy.fft
import math

#imports for Meliza 2014 CM data:
import os
from neo import AxonIO
import quantities as pq
# end of imports for Meliza 2014 CM data

random.seed(2022)
np.random.seed(2022)

# This code is an edit branching off of Python code "User Side Code Separate Train and Test Same Epoch Different CM Auto File Loop"
# on April 13, 2022, because I want to use this file to train on one data file and test on another, which the old code architecture
# cannot be easily changed to simultaneously accommodate the older purpose.

# In[2]:


# modify this
# neuron_txt_filename = 'Episode_2_voltage.txt'
epoch = None # also called "episode". set to None if not specified


# In[3]:


# ======== do not modify below ==========
neuron_name_list = ['32425a75',
                     '920061fe']
directories_list = ['CM_data/'+str(neuron_name_list[0])+'/',
                    'CM_data/'+str(neuron_name_list[1]+'/')
                    ]

do_not_use_list = ["2014_09_10_0001.abf",
                   "2014_09_10_0002.abf",
                   "2014_09_10_0003.abf"# file attached to Meliza's data when I received it said not to use these. The input signals are horrible
                   ] # bad data

# Change this value each time you want to test on a different data set
test_filename = "2014_09_10_0004.abf"

# Code for 2014 Meliza CM data
for neuron_directory_index in range(len(directories_list)):
    neuron_directory = directories_list[neuron_directory_index]
    for a_filename in os.listdir(neuron_directory):
        print("================================New File ==================================================")
        if a_filename in do_not_use_list:
            continue # skip this iteration if the filename is on the do_not_use_list
        if a_filename != "2014_09_10_0013.abf": #"2014_12_11_0017.abf":#"2014_09_10_0013.abf":#"2014_12_11_0017.abf":
            continue # want to try predicting only this neuron since it's complex and most interesting
        directory_of_test_VIt = directories_list[neuron_directory_index] + 'txt_V_I_t/'
        test_neuron_name =  neuron_name_list[neuron_directory_index]

        # Code for 2014 Meliza CM data
        # there is a constant junction potential offset that needs to be applied to the voltage
        junction_potential = pq.Quantity(11.6, 'mV')  # measured at 32 C (NOTE: I'm not completely sure if this applies to all measurements of CM in these directories. I should ask Prof. Meliza)
        # open the file
        does_this_exist =str(neuron_directory)
        print("Preparing to use file "+does_this_exist)
        fp = AxonIO(filename=neuron_directory+a_filename)
        # read the data. There is only one block in each file
        block = fp.read_block()
        # Neo calls sweeps "segments"
        sweep = block.segments[0]
        # each sweep (here, block.segments[0]) has one or more channels. Channel 0 is always V and channel 1 is always I.
        V = (sweep).analogsignals[0] - junction_potential
        I = (sweep).analogsignals[1]
        t = V.times - V.t_start #measured in seconds
        TT = ((t[1]-t[0])*1000).magnitude #this is milliseconds
        # print(V.magnitude.shape)
        # print(V)
        # print(I.magnitude.shape)
        # print(t.shape)
        V_and_I_arr = np.concatenate((V.magnitude, I.magnitude),axis=1)
        # print(V_and_I_arr.shape)
        t_arr_train = np.array([t]).transpose()

        fp_test = AxonIO(filename=directories_list[0]+test_filename)
        # read the data. There is only one block in each file
        block_test = fp_test.read_block()
        # Neo calls sweeps "segments"
        sweep_test = block_test.segments[0]
        V_test = (sweep_test).analogsignals[0] - junction_potential
        t_test = V_test.times - V_test.t_start #measured in seconds
        t_arr_test = np.array([t_test]).transpose()
        # make directory for storing data in .txt form if it doesn't exist yet
        directory_of_train_VIt = neuron_directory+'txt_V_I_t/'
        directory_to_store_txt_data = directory_of_train_VIt + str('test_data='+str(test_neuron_name)+'_'+str(test_filename[:-4])+'/')
        if not os.path.isdir(directory_to_store_txt_data):
            os.mkdir(directory_to_store_txt_data)
        if not os.path.isdir(directory_to_store_txt_data+ "training_and_prediction/"):
            os.mkdir(directory_to_store_txt_data+ "training_and_prediction/")

        np.savetxt(directory_to_store_txt_data + str(a_filename[:-4]) +"_VIt.txt", np.concatenate((V_and_I_arr, t_arr_train), axis=1))

        # neuron data/920061fe/2014_12_11_0017.abf from new Meliza data
        imported_data_train = np.loadtxt(directory_of_train_VIt + str(a_filename[:-4]) + "_VIt.txt")
        loaded_V_trainset = imported_data_train[:, 0]
        loaded_I_trainset = imported_data_train[:, 1]

        imported_data_test = np.loadtxt(directory_of_test_VIt + str(test_filename[:-4]) + "_VIt.txt")
        loaded_V_testset = imported_data_test[:, 0]
        loaded_I_testset = imported_data_test[:, 1]



        total_num_timesteps_in_data = len(loaded_V_trainset)
        train_timestep_end = round(total_num_timesteps_in_data*5.0/6.0)
        Voltage_train = loaded_V_trainset[:train_timestep_end]
        Current_train = loaded_I_trainset[:train_timestep_end]
        total_num_timesteps_in_testset =  len(loaded_V_testset)
        Voltage_test = loaded_V_testset[round(total_num_timesteps_in_testset*5.0/6.0):]
        Current_test = loaded_I_testset[round(total_num_timesteps_in_testset*5.0/6.0):]
        length = Voltage_train.shape[0]-1000 # - 1000 just to give breathing room
        PreLength = Voltage_test.shape[0]-1000 # - 1000 just to give breathing room

        # make directory to save plots to if it doesn't yet exist
        directory_to_store_plots = "plots/"+neuron_directory+str(a_filename[:-4])+"/test_data="+str(test_neuron_name)+"_"+str(test_filename[:-4])+"/"
        if not os.path.isdir(directory_to_store_plots):
            os.mkdir(directory_to_store_plots)

        # ===============================  POWER SPECTRA  =====================================

        # # FFT Train
        # sampling_rate = 30.0
        # fourier_transform = np.fft.rfft(Current_train)
        # abs_fourier_transform = np.abs(fourier_transform)
        # power_spectrum = np.square(abs_fourier_transform)
        # frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
        #
        # # Training Current with xlim
        # plt.figure()
        # plt.plot(frequency, power_spectrum/np.max(np.abs(power_spectrum)))
        # plt.title("Power(freq) of current_train (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+")")
        # plt.xlim((0,0.2))
        # plt.xlabel("Frequency (Scaling unknown)")
        # plt.ylabel("Normalized Power (1.0 = max from whole spectrum)")
        # plt.savefig(directory_to_store_plots+"Power spectrum of training current (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+").png")
        # plt.show()
        #
        # # Training Current with xlim
        # plt.figure()
        # plt.plot(frequency, power_spectrum/np.max(np.abs(power_spectrum)))
        # plt.title("Power(freq) of current_train (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+")")
        # plt.xlim((0,0.02))
        # plt.xlabel("Frequency (Scaling unknown)")
        # plt.ylabel("Normalized Power (1.0 = max from whole spectrum)")
        # plt.savefig(directory_to_store_plots+"Power spectrum of training current (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+")_ZoomedWayIn.png")
        # plt.show()
        #
        # # Training Current without xlim
        # plt.figure()
        # plt.plot(frequency, power_spectrum/np.max(np.abs(power_spectrum)))
        # plt.title("Power(freq) of current_train (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+")")
        # plt.xlim((0,1))
        # plt.xlabel("Frequency (Scaling unknown)")
        # plt.ylabel("Normalized Power (1.0 = max from whole spectrum)")
        # plt.savefig(directory_to_store_plots+"Power spectrum of training current (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+")_ZoomedOut.png")
        # plt.show()
        #
        # np.savetxt(directory_to_store_txt_data + str(a_filename[:-4]) + "_Fourier_Spectrum_training_current.txt",
        #            np.column_stack((frequency, power_spectrum)))

        # # Training Voltage with xlim
        # sampling_rate = 30.0
        # fourier_transform = np.fft.rfft(Voltage_train)
        # abs_fourier_transform = np.abs(fourier_transform)
        # power_spectrum = np.square(abs_fourier_transform)
        # frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
        # plt.figure()
        # plt.plot(frequency[10:], power_spectrum[10:]/np.max(np.abs(power_spectrum[10:])))
        # print(Voltage_train.shape)
        # print("THIS IS THE SHAPE")
        # plt.title("Power(freq) of voltage_train (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+")")
        # plt.xlim((0,0.2))
        # print(power_spectrum)
        # plt.xlabel("Frequency (Scaling unknown)")
        # plt.ylabel("Normalized Power (1.0 = max from [10:])")
        # plt.savefig(directory_to_store_plots+"Power spectrum of training voltage (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+").png")
        # plt.show()
        #
        # np.savetxt(directory_to_store_txt_data + str(a_filename[:-4]) + "_Fourier_Spectrum_training_voltage.txt",
        #            np.column_stack((frequency, power_spectrum)))
        #
        #
        # plt.figure()
        # plt.plot(frequency[10:], power_spectrum[10:]/np.max(np.abs(power_spectrum[10:])))
        # print(Voltage_train.shape)
        # print("THIS IS THE SHAPE")
        # plt.title("Power(freq) of voltage_train (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+")")
        # plt.xlim((0,0.02))
        # print(power_spectrum)
        # plt.xlabel("Frequency (Scaling unknown)")
        # plt.ylabel("Normalized Power (1.0 = max from [10:])")
        # plt.savefig(directory_to_store_plots+"Power spectrum of training voltage (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+")_ZoomedWayIn.png")
        # plt.show()
        #

        # # Training Voltage without xlim
        # sampling_rate = 30.0
        # fourier_transform = np.fft.rfft(Voltage_train)
        # abs_fourier_transform = np.abs(fourier_transform)
        # power_spectrum = np.square(abs_fourier_transform)
        # frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))
        # plt.figure()
        # plt.plot(frequency[10:], power_spectrum[10:] / np.max(np.abs(power_spectrum[10:])))
        # print(Voltage_train.shape)
        # print("THIS IS THE SHAPE")
        # plt.title("Power(freq) of voltage_train (Neuron " + str(neuron_name_list[neuron_directory_index]) + '_' + str(
        #     a_filename[:-4]) + ")")
        # plt.xlim((0, 1))
        # print(power_spectrum)
        # plt.xlabel("Frequency (Scaling unknown)")
        # plt.ylabel("Normalized Power (1.0 = max from [10:])")
        # plt.savefig(directory_to_store_plots + "Power spectrum of training voltage (Neuron " + str(
        #     neuron_name_list[neuron_directory_index]) + '_' + str(a_filename[:-4]) + ")_ZoomedOut.png")
        # plt.show()
        #
        # # FFT Test
        # sampling_rate = 30.0
        # fourier_transform = np.fft.rfft(Current_test)
        # abs_fourier_transform = np.abs(fourier_transform)
        # power_spectrum = np.square(abs_fourier_transform)
        # frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
        #
        # # Testing Current with xlim
        # plt.figure()
        # plt.plot(frequency, power_spectrum/np.max(np.abs(power_spectrum)))
        # plt.title("Power(freq) of current_test (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+")")
        # plt.xlim((0,0.2))
        # plt.ylabel("Normalized Power (1.0 = max from whole spectrum)")
        # plt.xlabel("Frequency (Scaling unknown)")
        # plt.savefig(directory_to_store_plots+"Power spectrum of test current (Neuron "+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+").png")
        # plt.show()
        #
        # np.savetxt(directory_to_store_txt_data + str(a_filename[:-4]) + "_Fourier_Spectrum_testing_current.txt",
        #            np.column_stack((frequency, power_spectrum)))


        # ===============================  END OF POWER SPECTRA  =====================================


        # plt.figure()
        # plt.plot(loaded_V_1, color='blue')
        # plt.title("Training Voltage")
        # plt.ylabel("Voltage (mV)")
        # plt.xlabel("Timestep in file (each timestep = 0.02ms)")
        # plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 first half Test 1 second half"+" Training Voltage.png")
        # plt.show()

        plt.figure()
        plt.plot(Voltage_train, color='blue')
        plt.title("Training Voltage")
        plt.ylabel("Voltage (mV)")
        plt.xlabel("Timestep in file (each timestep = "+str(TT)+"ms)")
        plt.savefig(directory_to_store_plots+"Train 1 first half Test 1 second half"+" Training Voltage.png")
        plt.show()
        plt.figure()
        plt.plot(Current_train, color='blue')
        plt.title("Training Current")
        plt.ylabel("Current (pA)")
        plt.xlabel("Timestep in file (each timestep = "+str(TT)+"ms)")
        plt.savefig(directory_to_store_plots+"Train 1 first half Test 1 second half"+" Training Current.png")
        plt.show()

        plt.figure()
        plt.plot(Voltage_test, color='orange')
        plt.title("Test Voltage")
        plt.ylabel("Voltage (mV)")
        plt.xlabel("Timestep in file (each timestep "+str(TT)+"ms)")
        plt.savefig(directory_to_store_plots+"Train 1 first half Test 1 second half"+" Test Voltage.png")
        plt.show()
        plt.figure()
        plt.plot(Current_test, color='orange')
        plt.title("Test Current")
        plt.ylabel("Current (pA)")
        plt.xlabel("Timestep in file (each timestep "+str(TT)+"ms)")
        plt.savefig(directory_to_store_plots+"Train 1 first half Test 1 second half"+" Test Current.png")
        plt.show()

        print("DONE HERE")
        # In[4]:


        tau = 3
        D = 3
        for tau in range(2,10):
            for D in range(2,10):
                print("========================New tau and D combination ==================")
                if tau != 7 or D!=2:
                    continue  # want to try predicting only this neuron since it's complex and most interesting
                time_start = time.time()
                Xdata = Voltage_train
                print("Shape of Xdata is "+str(Xdata.shape))
                NoCenters_no_thresh = 500
                NoCenters_above_thresh = 50
                DDF = Gauss()
                # Combine centers above threshold with centers determined by kmeans
                Centers_k_means = DDF.KmeanCenter(Xdata,NoCenters_no_thresh,D,length,tau);
                time_k_centers_done = time.time()
                print("Time to find k centers: "+str(time_k_centers_done-time_start))
                temp_array = copy.deepcopy(Xdata)
                temp_array[temp_array<-50]=-100
                Centers_above_thresh = DDF.KmeanCenter(temp_array,NoCenters_above_thresh,D,length,tau);
                # Center = np.concatenate((Centers_k_means,Centers_above_thresh),axis=0)
                Center = Centers_k_means

                print("C_km:"+str(Centers_k_means.shape))
                # times_where_V_gt_n75 = train_t_ind_arr[V_Train_delayed[0]>=-40]
                # V_Train_delayed_add = V_Train_delayed[:,(times_where_V_gt_n75-train_t_ind_arr[0])] # important to subtract train_t_ind_arr[0] (time training index vs total time index of V delayed array)
                # Centers_above_thresh = DDF.KmeanCenter(Xdata[Xdata>-50],100,D,length,tau);


                print("temp_arry:"+str(temp_array))
                # plt.figure()
                # plt.plot(temp_array)
                # plt.show()
                print("C_at:"+str(Centers_above_thresh.shape))

                NoCenters = np.shape(Center)[0]
                print(NoCenters)
                print("Centers:"+str(Center.shape))
                np.savetxt('centers/Center '+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+'(D,NumCenters)='+str((D,NoCenters))+'.txt',Center)
                Center = np.loadtxt('centers/Center '+str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+'(D,NumCenters)='+str((D,NoCenters))+'.txt')
                # Center = np.load('Example/Centers_train25k_5k_D'+str(D)+'_tau'+str(tau)+'.npy')



                stim_train = Current_train
                Pdata = Voltage_test
                beta_arr,R_arr = [1e-3,1e-2,1e-1,1e0,1e1,1e2],[1e-3,1e-2,1e-1,1e0,1e1,1e2]#[10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1],[10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2,10e3,10e4]#[10e4,10e3,10e2,10e1]#[10e-5,10e-4,10e-3]#[10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1]#[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2],[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2]#,[10e-3,10e-2,10e-1,10e0,10e1,10e2,10e3]
                # print(beta_arr)
                # print("PRINTED BETA ARR")
                bias = 50 # should be larger than tau*(D-1) or something like that
                X = np.arange(bias,bias+PreLength*TT,TT)
                stim_test = Current_test
                time_preparing_to_run_beta_r = time.time()
                print("Time to reach right before beta_r loop: "+str(time_preparing_to_run_beta_r-time_start))

                for beta in beta_arr:
                    for R in R_arr:
                        if (not math.isclose(beta,1.0)) or (not math.isclose(R,0.01)):
                            continue
                        print("(beta, R) = " + str((beta,R)))
                        time_beta_r_start = time.time()
                        # beta = 10e0
                        # R = 10e-4
                        title = str(neuron_name_list[neuron_directory_index])+'_train_'+str(a_filename[:-4])+'test_'+str(test_filename[:-4])+' with '+str(TT)+' time step, D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train Time = '+str(length)+', Centers = '+str(NoCenters)+', tau = '+str(tau)
                        print(R)
                        print("Shape of Xdata is now "+str(Xdata.shape))
                        F = DDF.FuncApproxF(Xdata,length,Center,beta,R,D,stim_train,tau)
                        time_beta_r_trained = time.time()
                        print("Time to run one beta-r  training: " + str(time_beta_r_trained - time_beta_r_start))
                        time_beta_r_start_prediction = time.time()
                        PredValidation = DDF.PredictIntoTheFuture(F,PreLength,stim_test[bias-1:],Pdata[bias-1-(D-1)*tau:])
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
                        print("Done with "+str((neuron_directory)+str(neuron_name_list[neuron_directory_index])+str((tau,D,beta,R))))
                        time_beta_r_end = time.time()
                        print("Time to run one beta-r train plus prediction: " + str(time_beta_r_end - time_beta_r_start))

                        print("Saving training and testing data to files")
                        training_times = t_arr_train[:train_timestep_end - 1000 + tau * D] # most time-lagged variable V(t-tau*D) goes from data[0:length]. V(t) goes from data[tau*D:length+tau*D]
                        used_Voltage_train = loaded_V_trainset[:train_timestep_end - 1000 + tau * D]
                        used_Current_train = loaded_I_trainset[:train_timestep_end - 1000 + tau * D]
                        testing_times = (t_arr_test[round(total_num_timesteps_in_testset * 5.0 / 6.0):])[bias:bias + PreLength]
                        used_Voltage_test = Voltage_test[bias:bias + PreLength]
                        used_Current_test = Current_test[bias:bias + PreLength]
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
                        plt.plot(Voltage_train[:100000])
                        plt.plot(Pdata[0:180000])
                        plt.show()
                        plt.figure()
                        plt.plot(np.sort(DDF.W))
                        plt.title("RBF Coefficients (Sorted)")
                        plt.xlabel('Index (Sorted)')
                        plt.ylabel("RBF Coefficient Value")
                        plt.savefig(directory_to_store_plots + title+"_RBF_Coefficients_(Sorted).png")
                        np.savetxt(directory_to_store_txt_data + title + "_RBF_Coefficients_(Sorted).txt",
                                   np.sort(DDF.W))

                        plt.figure()
                        plt.plot(np.sort(Center[:, 0]))
                        plt.title("Centers")
                        plt.xlabel("Sorted centers index")
                        plt.ylabel("Voltage")
                        plt.savefig(directory_to_store_plots + title+"_Sorted_centers_vs_index.png")
                        np.savetxt(directory_to_store_txt_data + title + "_Sorted_centers_vs_index.txt",
                                   np.sort(Center[:, 0]))



                        plt.figure()
                        plt.scatter(Center[:, 0], DDF.W[:-1])
                        plt.title("Weight as a function of center voltage (unsure)")
                        plt.xlabel("Center voltage")
                        plt.ylabel("Weight (Coeff of RBF)")
                        plt.savefig(directory_to_store_plots + title+ "Weight(center_voltage).png")
                        np.savetxt(directory_to_store_txt_data + title + "_Weight(center_voltage).txt",
                                   np.column_stack((Center[:, 0], DDF.W[:-1])))


                        # save
                        np.savetxt(directory_to_store_txt_data + "training_and_prediction/" + str(a_filename[:-4]) + "_training_VIt_(Nc,tau,D,beta,R)="+str((NoCenters,tau, D, beta, R))+".txt",
                                   np.column_stack((used_Voltage_train, used_Current_train, training_times)))
                        np.savetxt(directory_to_store_txt_data + "training_and_prediction/" + str(test_filename[:-4]) + "_test_truth_VIt_(Nc,tau,D,beta,R)="+str((NoCenters,tau, D, beta, R))+".txt",
                                   np.column_stack((used_Voltage_test, used_Current_test, testing_times)))
                        np.savetxt(directory_to_store_txt_data + "training_and_prediction/" +"train_"+str(a_filename[:-4])+"_test_"+str(test_filename[:-4]) + "_test_prediction_VIt_(Nc,tau,D,beta,R)="+str((NoCenters,tau, D, beta, R))+".txt",
                                   np.column_stack((Voltage_pred, used_Current_test, testing_times)))








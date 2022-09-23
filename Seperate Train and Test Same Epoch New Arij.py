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
# on June 8, 2022, because I want to use this file to train on one file.

# In[2]:


# modify this
# neuron_txt_filename = 'Episode_2_voltage.txt'
epoch = None # also called "episode". set to None if not specified


# In[3]:


# ======== do not modify below ==========
neuron_name_list = ['Lilac 114',
                     'Lilac 242',
                    'Red 171']
directories_list = ['7-7-2022/'+str(neuron_name_list[0])+'/Neuron 1/',
                    '7-7-2022/'+str(neuron_name_list[0])+'/Neuron 2/',
                    '7-7-2022/'+str(neuron_name_list[1])+'/Neuron 1/',
                    '7-7-2022/'+str(neuron_name_list[1])+'/Neuron 2/',
                    '7-7-2022/'+str(neuron_name_list[1])+'/Neuron 3/',
                    '7-7-2022/'+str(neuron_name_list[1])+'/Neuron 4/',
                    '7-7-2022/'+str(neuron_name_list[2])+'/Neuron 1/',
                    '7-7-2022/'+str(neuron_name_list[2])+'/Neuron 2/'
                    ]

do_not_use_list = []

# Code for 2014 Meliza CM data
neuron_directory_index = 0

R = 2
beta = 2

neuron_directory = directories_list[neuron_directory_index]
for a_filename in os.listdir(neuron_directory):
    print("================================New File ==================================================")

    # Code for 2014 Meliza CM data
    # there is a constant junction potential offset that needs to be applied to the voltage
    #junction_potential = pq.Quantity(11.6, 'mV')  # measured at 32 C (NOTE: I'm not completely sure if this applies to all measurements of CM in these directories. I should ask Prof. Meliza)
    # open the file
    does_this_exist =str(neuron_directory)
    print("Preparing to use file "+does_this_exist)
    fp = np.loadtxt(neuron_directory+a_filename)
    # read the data. There is only one block in each file
    ##print(str(fp[0][1]))

    # block = fp # NEED TO FIND NEW WAY TO READ IN THIS DATA SCORCHED EARTH RECOMMENDED
    # Neo calls sweeps "segments"
    # sweep = block.segments[0]
    # each sweep (here, block.segments[0]) has one or more channels. Channel 0 is always V and channel 1 is always I.
    fp_tp = np.transpose(fp)
    V = fp_tp[0] # - junction_potential
    I = fp_tp[1]
    #t = V.times - V.t_start #measured in seconds
    TT = 1 # set to 1 since I dont use it and I dont want to get rid of all its use cases right now
    # print(V.magnitude.shape)
    # print(V)
    # print(I.magnitude.shape)
    # print(t.shape)
    V_and_I_arr = np.concatenate((V, I))
    # print(V_and_I_arr.shape)
    #t_arr = np.array([t]).transpose()
    t_arr = np.arange(0, len(V))
    # make directory for storing data in .txt form if it doesn't exist yet
    directory_to_store_txt_data = neuron_directory+'txt_V_I_t/'
    if not os.path.isdir(directory_to_store_txt_data):
        os.mkdir(directory_to_store_txt_data)

    np.savetxt(directory_to_store_txt_data+str(a_filename[:-4])+"_VIt.txt", V_and_I_arr)

    # neuron data/920061fe/2014_12_11_0017.abf from new Meliza data
    # imported_data = np.loadtxt(directory_to_store_txt_data+str(a_filename[:-4])+"_VIt.txt")
    loaded_V = V
    loaded_I = I


    total_num_timesteps_in_data = len(loaded_V)
    train_timestep_end = round(total_num_timesteps_in_data*5.0/6.0)
    print("Train timestep end:"+str(train_timestep_end))
    print("Shape of loaded V:"+str(loaded_V.shape))
    Voltage_train = loaded_V[:train_timestep_end]
    print("Shape of Voltage train is "+str(Voltage_train.shape))
    Current_train = loaded_I[:train_timestep_end]
    Voltage_test = loaded_V[train_timestep_end:total_num_timesteps_in_data]
    Current_test = loaded_I[train_timestep_end:total_num_timesteps_in_data]
    length = Voltage_train.shape[0]-1000 # - 1000 just to give breathing room
    PreLength = Voltage_test.shape[0]-1000 # - 1000 just to give breathing room

    # make directory to save plots to if it doesn't yet exist
    directory_to_store_plots = "Output/"+neuron_directory+str(a_filename[:-4])+"/"
    if not os.path.isdir(directory_to_store_plots):
        os.mkdir(directory_to_store_plots)

    # ===============================  POWER SPECTRA  =====================================

    # FFT Train
    sampling_rate = 1.0 #/float(t[2]-t[1])
    print("Sampling rate is "+str(sampling_rate))
    print("Length of V is " +str(np.shape(Voltage_train)))
    fourier_transform = np.fft.rfft(Current_train)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    print("Length of power_spectrum is " +str(np.shape(power_spectrum)))
    frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
    print("Number of frequencies plotted is " +str(np.shape(frequency)))
    delta_freq = frequency[3]-frequency[2]
    print("Frequency spacing is " +str(delta_freq))


    # In[4]:


    for tau in range(2,10):
        for D in range(2,10):
            print("========================New tau and D combination ==================")
            time_start = time.time()
            Xdata = Voltage_train
            print("Shape of Xdata is "+str(Xdata.shape))
            NoCenters_no_thresh = 500
            NoCenters_above_thresh = 50
            DDF = Gauss()
            # Combine centers above threshold with centers determined by kmeans
            print("doin the kmeans center thing")
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
            # plt.close()
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
                    title = str(neuron_name_list[neuron_directory_index])+'_'+str(a_filename[:-4])+' with D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train Time = '+str(length)+', Centers = '+str(NoCenters)+', tau = '+str(tau)
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
                    training_times =   t_arr[:train_timestep_end - 1000 + tau*D] # most time-lagged variable V(t-tau*D) goes from data[0:length]. V(t) goes from data[tau*D:length+tau*D]
                    used_Voltage_train = loaded_V[:train_timestep_end - 1000 + tau*D]
                    used_Current_train = loaded_I[:train_timestep_end - 1000 + tau*D]
                    testing_times = (t_arr[train_timestep_end:total_num_timesteps_in_data])[bias:bias + PreLength]
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

                    plt.close()


                    # Coefficient Plotting
                    # plt.figure()
                    # plt.plot(Voltage_train[:100000])
                    # plt.plot(Pdata[0:180000])
                    # plt.close()
                    plt.figure()
                    plt.plot(np.sort(DDF.W))
                    plt.title("RBF Coefficients (Sorted)")
                    plt.xlabel('Index (Sorted)')
                    plt.ylabel("RBF Coefficient Value")
                    plt.savefig(directory_to_store_plots + title+"_RBF_Coefficients_(Sorted).png")
                    np.savetxt(directory_to_store_txt_data + title + "_RBF_Coefficients_(Sorted).txt",
                               np.sort(DDF.W))
                    plt.close()

                    # plt.figure()
                    # plt.plot(Voltage_train[:100000])
                    # plt.plot(Pdata[0:180000])
                    # plt.close()
                    plt.figure()
                    plt.plot(DDF.W)
                    plt.title("RBF Coefficients (Unsorted)")
                    plt.xlabel('Index (Unsorted)')
                    plt.ylabel("RBF Coefficient Value")
                    plt.savefig(directory_to_store_plots + title+"_RBF_Coefficients_(Unsorted).png")
                    np.savetxt(directory_to_store_txt_data + title + "_RBF_Coefficients_(Unsorted).txt",
                               DDF.W)
                    plt.close()



                    plt.figure()
                    plt.plot(np.sort(Center[:, 0]))
                    plt.title("Centers")
                    plt.xlabel("Sorted centers index")
                    plt.ylabel("Voltage")
                    plt.savefig(directory_to_store_plots + title+"_Sorted_centers_vs_index.png")
                    np.savetxt(directory_to_store_txt_data + title + "_Sorted_centers_vs_index.txt",
                               np.sort(Center[:, 0]))
                    plt.close()



                    plt.figure()
                    plt.scatter(Center[:, 0], DDF.W[:-1])
                    plt.title("Weight as a function of center voltage (unsure)")
                    plt.xlabel("Center voltage")
                    plt.ylabel("Weight (Coeff of RBF)")
                    plt.savefig(directory_to_store_plots + title+ "Weight(center_voltage).png")
                    np.savetxt(directory_to_store_txt_data + title + "_Weight(center_voltage).txt",
                               np.column_stack((Center[:, 0], DDF.W[:-1])))
                    plt.close()

                    #
                    # # save
                    # np.savetxt(directory_to_store_txt_data + "training_and_prediction/" + str(a_filename[:-4]) + "_training_VIt_(Nc,tau,D,beta,R)="+str((NoCenters,tau, D, beta, R))+".txt",
                    #            np.column_stack((used_Voltage_train, used_Current_train, training_times)))
                    # np.savetxt(directory_to_store_txt_data + "training_and_prediction/" + str(a_filename[:-4]) + "_test_truth_VIt_(Nc,tau,D,beta,R)="+str((NoCenters,tau, D, beta, R))+".txt",
                    #            np.column_stack((used_Voltage_test, used_Current_test, testing_times)))
                    # np.savetxt(directory_to_store_txt_data + "training_and_prediction/" + str(a_filename[:-4]) + "_test_prediction_VIt_(Nc,tau,D,beta,R)="+str((NoCenters,tau, D, beta, R))+".txt",
                    #            np.column_stack((Voltage_pred, used_Current_test, testing_times)))








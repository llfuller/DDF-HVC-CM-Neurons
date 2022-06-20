#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TimeDelay_Neuron_DDF_GaussianForm import *
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import copy
random.seed(2022)
np.random.seed(2022)


# In[2]:


# modify this
# use_neuron_index =
use_neuron_index = 1
# neuron_txt_filename = 'Episode_2_voltage.txt'
epoch = None # also called "episode". set to None if not specified


# In[3]:


# ======== do not modify below ==========
neuron_name_list = ['Neuron_52_6-8-16',
                    'Neuron_57_6-8-16',
                    'Neuron_61_6-8-16',
                    'Neuron_8-15-2019']
directories_list = ['HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 52/',
                   'HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 57/',
                   'HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 61/',
                   'HVC_ra_x_i_data_2016_2019/08-15-2019/']

#neuron 2019
# use_neuron_index = 3

#neuron 52 from 2016 (HVC_RA)
use_neuron_index = 0
# train 1, 14, test 16 (34 looks same as 1)
loaded_V_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_voltage.txt')
loaded_I_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_current.txt')

Voltage_train = loaded_V_1[:60000]
Current_train = loaded_I_1[:60000]
Voltage_test = loaded_V_1[60000:120000]
Current_test = loaded_I_1[60000:120000]
length = Voltage_train.shape[0]-1000
PreLength = Voltage_test.shape[0]-1000
print(Voltage_test.shape)
#neuron 57 from 2016 (HVC_x)
# use_neuron_index = 1

# still neuron 57 from 2016 (HVC-x)


#neuron 61 from 2016 (HVC_I)
# use_neuron_index = 2

print("Here 1")
print(np.shape(Voltage_train))
print(np.shape(Current_train))
print(np.shape(Current_test))
# Voltage_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00014_voltage.txt')
# Current_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00014_current.txt')
print("Here 2")
print(np.shape(Voltage_train))
print(np.shape(Current_train))


plt.figure()
plt.plot(Voltage_train, color='blue')
# plt.xlim((30000,50000))
plt.title("Training Voltage")
plt.ylabel("Voltage (mV)")
plt.xlabel("Timestep in file (each timestep = 0.02ms)")
# plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 14 Test 16"+" Training Voltage")
plt.show()
plt.figure()
plt.plot(Current_train, color='blue')
plt.xlim((30000,50000))
plt.title("Training Current")
plt.ylabel("Current")
plt.xlabel("Timestep in file (each timestep = 0.02ms)")
# plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 14 Test 16"+" Training Current")
plt.show()

plt.figure()
plt.plot(Voltage_test, color='orange')
# plt.xlim((30000,50000))
plt.title("Test Voltage")
plt.ylabel("Voltage (mV)")
plt.xlabel("Timestep in file (each timestep = 0.02ms)")
# plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 14 Test 16"+" Test Voltage")
plt.show()
plt.figure()
plt.plot(Current_test, color='orange')
plt.xlim((10000,60000))
plt.title("Test Current")
plt.ylabel("Current")
plt.xlabel("Timestep in file (each timestep = 0.02ms)")
# plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 14 Test 16"+" Test Current")
plt.show()

print("DONE HERE")
# In[4]:


tau = 3
D = 3
for tau in range(2,10):
    for D in range(2,10):
        Xdata = Voltage_train
        TT = 0.02
        NoCenters_no_thresh = 1500
        NoCenters_above_thresh = 50
        DDF = Gauss()
        # Combine centers above threshold with centers determined by kmeans
        Centers_k_means = DDF.KmeanCenter(Xdata,NoCenters_no_thresh,D,length,tau);
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
        print("C_at:"+str(Centers_above_thresh.shape))

        NoCenters = np.shape(Center)[0]
        print(NoCenters)
        print("Centers:"+str(Center.shape))
        np.savetxt('centers/Center '+neuron_name_list[use_neuron_index]+'(D,NumCenters)='+str((D,NoCenters))+'.txt',Center)
        Center = np.loadtxt('centers/Center '+neuron_name_list[use_neuron_index]+'(D,NumCenters)='+str((D,NoCenters))+'.txt')
        # Center = np.load('Example/Centers_train25k_5k_D'+str(D)+'_tau'+str(tau)+'.npy')

        stim_train = Current_train
        Pdata = Voltage_test
        beta_arr,R_arr = [10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1],[10e-5,10e-4,10e-3]#[10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1]#[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2],[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2]#,[10e-3,10e-2,10e-1,10e0,10e1,10e2,10e3]
        bias = 50 # should be larger than tau*(D-1) or something like that
        X = np.arange(bias,bias+PreLength*TT,TT)
        stim_test = Current_test

        for beta in beta_arr:
            for R in R_arr:
                # beta = 10e0
                # R = 10e-4
                title = neuron_name_list[use_neuron_index]+' with 0.02 time step, D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train Time = '+str(length)+', Centers = '+str(NoCenters)+', tau = '+str(tau)
                print(R)
                F = DDF.FuncApproxF(Xdata,length,Center,beta,R,D,stim_train,tau)
                PredValidation = DDF.PredictIntoTheFuture(F,PreLength,stim_test[bias-1:],Pdata[bias-1-(D-1)*tau:])
                # Tau8
                plt.figure(figsize=(20,10))
                plt.plot(X,Pdata[bias:bias + PreLength],label = 'True Voltage', color = 'black')
                plt.plot(X,PredValidation[tau*(D-1)+1:tau*(D-1)+PreLength+1],label = 'Prediction', color = 'r')
                plt.xlabel('time (ms)',fontsize=20)
                plt.ylabel('Voltage',fontsize=20)
                plt.legend()
                plt.title(title,fontsize=20)
                #plt.savefig('Validation Prediction Result')
                plt.savefig(title+'.png')
                # plt.show



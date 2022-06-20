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

#imports for Meliza 2014 CM data:
import os
from neo import AxonIO
import quantities as pq
# end of imports for Meliza 2014 CM data

random.seed(2022)
np.random.seed(2022)


# In[2]:


# modify this
# neuron_txt_filename = 'Episode_2_voltage.txt'
epoch = None # also called "episode". set to None if not specified


# In[3]:


# ======== do not modify below ==========
neuron_name_list = ['Neuron_52_6-8-16',
                    'Neuron_57_6-8-16',
                    'Neuron_61_6-8-16',
                    'Neuron_8-15-2019',
                    'CM_2014_12_11_0017_Segment0' #920061fe
                    ]
directories_list = ['HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 52/',
                   'HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 57/',
                   'HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 61/',
                   'HVC_ra_x_i_data_2016_2019/08-15-2019/',
                    'CM_data/']

#neuron 2019
# use_neuron_index = 3

#neuron 52 from 2016 (HVC_RA)
# TT = 0.02
# use_neuron_index = 0
# # train 1, 14, test 16 (34 looks same as 1)
# loaded_V_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_voltage.txt')
# loaded_I_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_current.txt')
#
# Voltage_train = loaded_V_1[:60000]
# Current_train = loaded_I_1[:60000]
# Voltage_test = loaded_V_1[60000:120000]
# Current_test = loaded_I_1[60000:120000]
# length = Voltage_train.shape[0]-1000
# PreLength = Voltage_test.shape[0]-1000

#neuron 57 from 2016 (HVC_x)
# TT = 0.02
# use_neuron_index = 1
# loaded_V_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00019_voltage.txt')
# loaded_I_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00019_current.txt')
#
# Voltage_train = loaded_V_1[:85000]
# Current_train = loaded_I_1[:85000]
# Voltage_test = loaded_V_1[85000:150000]
# Current_test = loaded_I_1[85000:150000]
# length = Voltage_train.shape[0]-1000
# PreLength = Voltage_test.shape[0]-1000

#neuron 61 from 2016 (HVC_I)
# TT = 0.02
# use_neuron_index = 2
# loaded_V_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0002_voltage.txt')
# loaded_I_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0002_current.txt')
#
# Voltage_train = loaded_V_1[:70000]
# Current_train = loaded_I_1[:70000]
# Voltage_test = loaded_V_1[70000:160000]
# Current_test = loaded_I_1[70000:160000]
# length = Voltage_train.shape[0]-1000
# PreLength = Voltage_test.shape[0]-1000

# still neuron 61 from 2016
# loaded_V_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_voltage.txt')
# loaded_I_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_current.txt')
#
# Voltage_train = loaded_V_1[:60000]
# Current_train = loaded_I_1[:60000]
# Voltage_test = loaded_V_1[60000:120000]
# Current_test = loaded_I_1[60000:120000]
# length = Voltage_train.shape[0]-1000
# PreLength = Voltage_test.shape[0]-1000

# Code for 2014 Meliza CM data
TT = 0.025
use_neuron_index = 4
# there is a constant junction potential offset that needs to be applied to the voltage
junction_potential = pq.Quantity(11.6, 'mV')  # measured at 32 C
# open the file
fp = AxonIO(filename="cm_ddf/data/920061fe/2014_12_11_0017.abf")
# read the data. There is only one block in each file
block = fp.read_block()
# Neo calls sweeps "segments"
sweep = block.segments[2]
# each sweep (here, block.segments[n]) has one or more channels. Channel 0 is always V and channel 1 is always I.
V = (block.segments[0]).analogsignals[0] - junction_potential
I = (block.segments[0]).analogsignals[1]
t = V.times - V.t_start
print(V.magnitude.shape)
print(V)
print(I.magnitude.shape)
print(t.shape)
V_and_I_arr = np.concatenate((V.magnitude, I.magnitude),axis=1)
print(V_and_I_arr.shape)
t_arr = np.array([t]).transpose()
np.savetxt(directories_list[use_neuron_index]+"Meliza_CM_920061fe_2014_12_11_0017_Segment_0_VIt.txt", np.concatenate((V_and_I_arr, t_arr),axis=1))

# neuron data/920061fe/2014_12_11_0017.abf from new Meliza data
imported_data = np.loadtxt(directories_list[use_neuron_index]+'Meliza_CM_920061fe_2014_12_11_0017_Segment_0_VIt.txt')
loaded_V = imported_data[:,0]
loaded_I = imported_data[:,1]


print(loaded_V)

Voltage_train = loaded_V[:500000]
Current_train = loaded_I[:500000]
Voltage_test = loaded_V[500000:600000]
Current_test = loaded_I[500000:600000]
length = Voltage_train.shape[0]-1000
PreLength = Voltage_test.shape[0]-1000

# FFT Train
sampling_rate = 30.0
fourier_transform = np.fft.rfft(Current_train)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
plt.figure()
plt.plot(frequency, power_spectrum)
plt.title("Power spectrum of training current (Neuron "+str(neuron_name_list[use_neuron_index]))
plt.xlim((0,1.5))
plt.xlabel("Frequency (Scaling unknown)")
plt.ylabel("Power")
plt.savefig("Power spectrum of training current (Neuron "+str(neuron_name_list[use_neuron_index])+".png")
plt.show()

# FFT Test
sampling_rate = 30.0
fourier_transform = np.fft.rfft(Current_test)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
plt.figure()
plt.plot(frequency, power_spectrum)
plt.title("Power spectrum of test current (Neuron "+str(neuron_name_list[use_neuron_index]))
plt.xlim((0,1.5))
plt.ylabel("Power")
plt.xlabel("Frequency (Scaling unknown)")
plt.savefig("Power spectrum of test current (Neuron "+str(neuron_name_list[use_neuron_index])+".png")
plt.show()




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
plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 first half Test 1 second half"+" Training Voltage.png")
plt.show()
plt.figure()
plt.plot(Current_train, color='blue')
plt.title("Training Current")
plt.ylabel("Current")
plt.xlabel("Timestep in file (each timestep = "+str(TT)+"ms)")
plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 first half Test 1 second half"+" Training Current.png")
plt.show()

plt.figure()
plt.plot(Voltage_test, color='orange')
plt.title("Test Voltage")
plt.ylabel("Voltage (mV)")
plt.xlabel("Timestep in file (each timestep "+str(TT)+"ms)")
plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 first half Test 1 second half"+" Test Voltage.png")
plt.show()
plt.figure()
plt.plot(Current_test, color='orange')
plt.title("Test Current")
plt.ylabel("Current")
plt.xlabel("Timestep in file (each timestep "+str(TT)+"ms)")
plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 first half Test 1 second half"+" Test Current.png")
plt.show()

print("DONE HERE")
# In[4]:


tau = 3
D = 3
for tau in range(2,10):
    for D in range(2,10):
        Xdata = Voltage_train
        NoCenters_no_thresh = 500
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
        # plt.figure()
        # plt.plot(temp_array)
        # plt.show()
        print("C_at:"+str(Centers_above_thresh.shape))

        NoCenters = np.shape(Center)[0]
        print(NoCenters)
        print("Centers:"+str(Center.shape))
        np.savetxt('centers/Center '+neuron_name_list[use_neuron_index]+'(D,NumCenters)='+str((D,NoCenters))+'.txt',Center)
        Center = np.loadtxt('centers/Center '+neuron_name_list[use_neuron_index]+'(D,NumCenters)='+str((D,NoCenters))+'.txt')
        # Center = np.load('Example/Centers_train25k_5k_D'+str(D)+'_tau'+str(tau)+'.npy')


        # # In[5]:
        #
        #
        # plt.show()
        # plt.figure()
        # plt.plot(np.sort(Center[:,0]))
        # plt.title("Voltage of centers")
        # plt.show()
        #
        #
        # # In[6]:
        #
        #
        # plt.show()
        # plt.figure()
        # plt.plot(np.sort(Center[:,0]))
        # plt.title("Voltage of centers")
        # plt.show()
        #
        # plt.show()
        # plt.figure()
        # plt.plot(Voltage_train[10550:11200])
        # plt.title("Demonstration of training voltage resolution around spikes")
        # plt.show()


        # In[7]:


        stim_train = Current_train
        Pdata = Voltage_test
        beta_arr,R_arr = [10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1],[10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2,10e3,10e4]#[10e4,10e3,10e2,10e1]#[10e-5,10e-4,10e-3]#[10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1]#[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2],[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2]#,[10e-3,10e-2,10e-1,10e0,10e1,10e2,10e3]
        print(beta_arr)
        print("PRINTED BETA ARR")
        bias = 50 # should be larger than tau*(D-1) or something like that
        X = np.arange(bias,bias+PreLength*TT,TT)
        stim_test = Current_test


        # In[8]:



        # In[9]:


        for beta in beta_arr:
            for R in R_arr:
                # beta = 10e0
                # R = 10e-4
                title = neuron_name_list[use_neuron_index]+' with '+str(TT)+' time step, D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train Time = '+str(length)+', Centers = '+str(NoCenters)+', tau = '+str(tau)
                print(R)
                F = DDF.FuncApproxF(Xdata,length,Center,beta,R,D,stim_train,tau)
                PredValidation = DDF.PredictIntoTheFuture(F,PreLength,stim_test[bias-1:],Pdata[bias-1-(D-1)*tau:])
                # Tau8
                plt.figure(figsize=(20,10))
                plt.plot(X,Pdata[bias:bias + PreLength],label = 'True Voltage', color = 'black')
                plt.plot(X,PredValidation[tau*(D-1)+1:tau*(D-1)+PreLength+1],'r--',label = 'Prediction')
                plt.xlabel('time (ms)',fontsize=20)
                plt.ylabel('Voltage',fontsize=20)
                plt.legend()
                plt.title(title,fontsize=20)
                #plt.savefig('Validation Prediction Result')
                plt.savefig(title+'.png')
                # plt.show


        # In[10]:


        # beta = 10e2
        # R = 10e-3
        # title = neuron_name_list[use_neuron_index]+' with 0.02 time step, D = 3, Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train Time = 50000, Centers = 1000, tau = '+str(tau)
        # print(R)
        # F = DDF.FuncApproxF(Xdata,length,Center,beta,R,D,stim_train,tau)
        # PredValidation = DDF.PredictIntoTheFuture(F,PreLength,stim_test[bias-1:],Pdata[bias-1-(D-1)*tau:])
        # # Tau8
        # plt.figure(figsize=(20,10))
        # plt.plot(X,Pdata[bias:bias + PreLength],label = 'True Voltage', color = 'black')
        # plt.plot(X,PredValidation[tau*(D-1)+1:tau*(D-1)+PreLength+1],label = 'Prediction', color = 'r')
        # plt.xlabel('time (ms)',fontsize=20)
        # plt.ylabel('Voltage',fontsize=20)
        # plt.legend()
        # plt.title(title,fontsize=20)
        # #plt.savefig('Validation Prediction Result')
        # plt.savefig(title+'.png')
        # plt.show


        # In[11]:


        # plt.plot(Voltage_train[:100000])
        # plt.plot(Pdata[0:180000])
        # plt.show()
        # plt.figure()
        # plt.plot(np.sort(DDF.W))
        # plt.title("RBF Coefficients (Sorted)")
        # plt.xlabel('Index (Sorted)')
        # plt.ylabel("RBF Coefficient Value")
        # plt.show()
        # plt.figure()
        # plt.plot(np.sort(Center[:,0]))
        # plt.title("Centers")
        # plt.xlabel("Sorted centers index")
        # plt.ylabel("Voltage")
        # plt.show()
        #
        # # plt.show()
        # # plt.figure()
        # # color_vals = (np.roll(Center,-1)-Center)[:,0]
        # # plt.title("Sorted color values")
        # # color_vals[color_vals<-20] = 0
        # # color_vals[color_vals>5] = 0
        # # plt.plot(np.sort(color_vals))
        # # plt.show()
        #
        # plt.scatter(Center[:,0],DDF.W[:-1])
        # plt.title("Weight as a function of center voltage (unsure)")
        # plt.xlabel("Center voltage")
        # plt.ylabel("Weight (Coeff of RBF)")
        # plt.show()
        #
        #
        # # In[12]:
        #
        #
        # plt.plot(Current_train[:100000])
        # plt.plot(Current_test[0:180000])
        #
        #
        # # In[13]:
        #
        #
        # print("{:.1e}".format(15.000002))


        # In[ ]:





        # In[ ]:





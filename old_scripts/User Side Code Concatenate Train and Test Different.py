#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TimeDelay_Neuron_DDF_GaussianForm import *
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import copy
random.seed()
np.random.seed()


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
# Voltage_train = np.loadtxt(directories_list[use_neuron_index]+'Episode_2_voltage.txt')
# Current_train = np.loadtxt(directories_list[use_neuron_index]+'Episode_2_current.txt')
# Voltage_test = np.loadtxt(directories_list[use_neuron_index]+'Episode_2_voltage.txt')
# Current_test = np.loadtxt(directories_list[use_neuron_index]+'Episode_2_current.txt')

#neuron 52 from 2016 (HVC_RA)
# episode 2
# length = 110000
# PreLength = 140000
# Voltage_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0002_voltage.txt')
# Current_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0002_current.txt')
# Voltage_test = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0002_voltage.txt')
# Current_test = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0002_current.txt')
# episode 14
# length = 110000
# PreLength = 140000
# Voltage_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00014_voltage.txt')
# Current_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00014_current.txt')

#neuron 57 from 2016 (HVC_x)
# concatenate episodes 23 and 24, test on 28.
V_23_arr_len_cutoff = 105000
V_24_arr_len_cutoff = 105000
V_28_arr_len_cutoff = 140000
length = V_23_arr_len_cutoff + V_24_arr_len_cutoff-1000
PreLength = V_28_arr_len_cutoff - 1000
loaded_V_23 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00023_voltage.txt')[:V_23_arr_len_cutoff]
loaded_I_23 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00023_current.txt')[:V_23_arr_len_cutoff]
loaded_V_24 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00024_voltage.txt')[:V_24_arr_len_cutoff]
loaded_I_24 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00024_current.txt')[:V_24_arr_len_cutoff]
# loaded_V_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00029_voltage.txt')[:V_29_arr_len_cutoff]
# loaded_I_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00029_current.txt')[:V_29_arr_len_cutoff]
loaded_V_28 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00028_voltage.txt')[:V_28_arr_len_cutoff]
loaded_I_28 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00028_current.txt')[:V_28_arr_len_cutoff]

# still neuron 57 from 2016 (HVC-x)
# concatenate episodes 23 and 24, test on 28.
V_19_arr_len_cutoff = 105000
V_20_arr_len_cutoff = 105000
V_23_arr_len_cutoff = 105000
V_24_arr_len_cutoff = 105000
V_29_arr_len_cutoff = 105000
V_28_arr_len_cutoff = 140000
length = V_23_arr_len_cutoff + V_24_arr_len_cutoff-1000
PreLength = V_28_arr_len_cutoff - 1000
loaded_V_19 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00019_voltage.txt')[:V_19_arr_len_cutoff]
loaded_I_19 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00019_current.txt')[:V_19_arr_len_cutoff]
loaded_V_20 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00020_voltage.txt')[:V_20_arr_len_cutoff]
loaded_I_20 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00020_current.txt')[:V_20_arr_len_cutoff]
loaded_V_23 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00023_voltage.txt')[:V_23_arr_len_cutoff]
loaded_I_23 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00023_current.txt')[:V_23_arr_len_cutoff]
loaded_V_24 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00024_voltage.txt')[:V_24_arr_len_cutoff]
loaded_I_24 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00024_current.txt')[:V_24_arr_len_cutoff]
loaded_V_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00029_voltage.txt')[:V_29_arr_len_cutoff]
loaded_I_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00029_current.txt')[:V_29_arr_len_cutoff]

# loaded_V_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00029_voltage.txt')[:V_29_arr_len_cutoff]
# loaded_I_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00029_current.txt')[:V_29_arr_len_cutoff]
loaded_V_28 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00028_voltage.txt')[:V_28_arr_len_cutoff]
loaded_I_28 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00028_current.txt')[:V_28_arr_len_cutoff]


Voltage_train = np.concatenate((loaded_V_19, loaded_V_20, loaded_V_23, loaded_V_24, loaded_V_29))
Current_train = np.concatenate((loaded_I_19, loaded_I_20, loaded_I_23, loaded_I_24, loaded_I_29))
Voltage_test = loaded_V_28
Current_test = loaded_I_28
# print(np.shape(loaded_I_1))
# print(np.shape(loaded_I_19))
# print(np.shape(loaded_I_22))


#neuron 61 from 2016 (HVC_I)
# episode 1
# length = 110000
# PreLength = 140000
# Voltage_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_voltage.txt')
# Current_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_current.txt')
#episode 19
# length = 110000
# PreLength = 140000
# Voltage_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_voltage.txt')
# Current_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_current.txt')

# concatenate episodes 1 and 19, test on 29.
# V_1_arr_len_cutoff = 105000
# V_19_arr_len_cutoff = 105000
# V_29_arr_len_cutoff = 140000
# length = V_1_arr_len_cutoff + V_19_arr_len_cutoff-1000
# PreLength = V_29_arr_len_cutoff - 1000
# loaded_V_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_voltage.txt')[:V_1_arr_len_cutoff]
# loaded_I_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_current.txt')[:V_1_arr_len_cutoff]
# loaded_V_19 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00019_voltage.txt')[:V_19_arr_len_cutoff]
# loaded_I_19 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00019_current.txt')[:V_19_arr_len_cutoff]
# # loaded_V_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00029_voltage.txt')[:V_29_arr_len_cutoff]
# # loaded_I_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00029_current.txt')[:V_29_arr_len_cutoff]
# loaded_V_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00048_voltage.txt')[:V_29_arr_len_cutoff]
# loaded_I_29 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00048_current.txt')[:V_29_arr_len_cutoff]
#
# Voltage_train = np.concatenate((loaded_V_1, loaded_V_19))
# Current_train = np.concatenate((loaded_I_1, loaded_I_19))
# Voltage_test = loaded_V_29
# Current_test = loaded_I_29
# print(np.shape(loaded_I_1))
# print(np.shape(loaded_I_19))
# print(np.shape(loaded_I_22))
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
plt.plot(Voltage_test, color='orange')
plt.show()
plt.figure()
plt.plot(Current_train, color='blue')
plt.plot(Current_test, color='orange')
plt.show()

print("DONE HERE")
# In[4]:


tau = 3
D = 3
for tau in range(2,10):
    for D in range(2,10):
        Xdata = Voltage_train
        TT = 0.02
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
        beta_arr,R_arr = [10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1],[10e-5,10e-4,10e-3]#[10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1]#[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2],[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2]#,[10e-3,10e-2,10e-1,10e0,10e1,10e2,10e3]
        bias = 50 # should be larger than tau*(D-1) or something like that
        X = np.arange(bias,bias+PreLength*TT,TT)
        stim_test = Current_test


        # In[8]:



        # In[9]:


        for beta in beta_arr:
            for R in R_arr:
                # beta = 10e0
                # R = 10e-4
                title = neuron_name_list[use_neuron_index]+' with 0.02 time step, D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train Time = 50000, Centers = '+str(NoCenters)+', tau = '+str(tau)
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





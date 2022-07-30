#!/usr/bin/env python
# coding: utf-8

# In[1]:



from TimeDelay_Neuron_DDF_GaussianForm_LPS_early_version import *
from Fourier_Power_Spectrum import *
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

θ=[-61.93402836804239, 0.053999984513550134, 0.8677675132377433, 0.21094758823620002, 0.8673893800823501, 0.06463106009370051, 0.5368278680702148, 0.034801309766394395, 0.2629320245028402, 0.18635713309843333, 0.992863241496597, 0.9879736378591375, 0.5639252977789222, 0.7703557770831406, 0.3999940134854983, 150.65112731743588, 8.925651590974955, 49.086890011293676, 124.46183364682842, 0.7326262126952293, 1.5787349674992859, 3.7808472023523008, -66.22008947137167, 0.012024239193339614, -105.61790029997763, 0.6471293077332771, 8.08863953487089, 0.20724216148907304, 0.2728691459839464, 14.531212521513075, -0.054064741667474436, -22.78305852040714, -0.7153235615665068, -15.533841412033809, -86.08017380269727, -8.534653707506266, -0.11176514742260224, -43.4579767881865, 28.690779089823195, -0.00012204864425768847, -99.99664591194461, 31.775823913365354, -24.78067041995729, 6.005340205277638, 8.63823861520788, -10.087948987067847, 10.862059490607107, 33.81870582735651, -22.094680223135065, 10.147666479022288, 55.10854255397067, -28.051128176925005, 8.291500889922348, 25.575861938775848, 13.759408712422493, 11.388013273642223, -46.93101265686462, 16.32587595005947, 93.31552392777911, 37.665821487984786, 29.67668073324702, 9.580115826685166, 20.18556370676597, 0.02354327054488278, 0.19755436289318964, 0.14056010764790958, 0.6789422014322583, 176.39691337734806, 0.020257220783194835, 1.2471745968859276, 53.28620013766909, 0.16081583040084538, 0.030454186084830095, 1.8974888113214765, 0.06494438535366333, 0.5733558157973574, 0.5682582149312141, 0.1481098706558182, 103.43239161540912, 7.265638983689308, 4.057076960728887, 79.23928255912374, 33.06275563454173, 42.99153018700545, 197.689392415391, 914.1897391966728, 3.2050352920928056, 46.321464493541086]



#neuron 2019
# use_neuron_index = 3
for lmn in range(1, 5):
    for ijk in θ[60:71]:
        if (ijk != 0):
            #neuron 52 from 2016 (HVC_RA)
            use_neuron_index = 0
            # train 1, 14, test 16 (34 looks same as 1)
            loaded_V_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_voltage.txt')
            loaded_I_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_current.txt')

            fc_tau = ijk
            print("lpf tau = "+str(fc_tau)+"\n")

            FPS_Function_Output_List = calc_Fourier_power_spectrum(np.array([loaded_I_1, loaded_V_1]), np.array([[i * (1 / 50000)] for i in range(len(loaded_I_1))]))

            h_final = 0

            frequency_list = FPS_Function_Output_List[1]

            for h in range(len(frequency_list)):
                if frequency_list[h] > 0 and fc_tau >= 1.0/(2.0 * np.pi * frequency_list[h]):
                    fc = frequency_list[h] / frequency_list[len(frequency_list) - 1]
                    h_final = h
                    print("lpf cutoff frequency = " + str(frequency_list[h]) + "Hz, at entry number " + str(h) + " with a tau of " + str(1 / (2 * np.pi * frequency_list[h])) + "\n")
                    break

            b = lmn * 0.001 #transition band
            N = int(np.ceil((4 / b)))

            if not N % 2: N += 1
            n = np.arange(N)

            sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
            window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
            sinc_func = sinc_func * window
            sinc_func = sinc_func / np.sum(sinc_func)

            Current_train = np.convolve(loaded_I_1, sinc_func, "same")[:60000] #loaded_I_1[:60000] #
            Current_test = np.convolve(loaded_I_1, sinc_func, "same")[60000:120000] #loaded_I_1[60000:120000] #

            Voltage_train = loaded_V_1[:60000]
            Voltage_test = loaded_V_1[60000:120000]
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
            plt.close()
            plt.figure()
            plt.plot(Current_train, color='blue')
            plt.xlim((30000,50000))
            plt.title("Training Current")
            plt.ylabel("Current")
            plt.xlabel("Timestep in file (each timestep = 0.02ms)")
            # plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 14 Test 16"+" Training Current")
            plt.close()

            plt.figure()
            plt.plot(Voltage_test, color='orange')
            # plt.xlim((30000,50000))
            plt.title("Test Voltage")
            plt.ylabel("Voltage (mV)")
            plt.xlabel("Timestep in file (each timestep = 0.02ms)")
            # plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 14 Test 16"+" Test Voltage")
            plt.close()
            plt.figure()
            plt.plot(Current_test, color='orange')
            plt.xlim((10000,60000))
            plt.title("Test Current")
            plt.ylabel("Current")
            plt.xlabel("Timestep in file (each timestep = 0.02ms)")
            # plt.savefig(str(neuron_name_list[use_neuron_index])+"Train 1 14 Test 16"+" Test Current")
            plt.close()

            print("DONE HERE")
            # In[4]:


            #tau = 3
            #D = 3
            for tau in range(5, 10):
                for D in range(8,12):
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
                    print("C_at:"+str(Centers_above_thresh.shape))

                    NoCenters = np.shape(Center)[0]
                    print(NoCenters)
                    print("Centers:"+str(Center.shape))
                    np.savetxt('centers/Center '+neuron_name_list[use_neuron_index]+'(D,NumCenters)='+str((D,NoCenters))+'.txt',Center)
                    Center = np.loadtxt('centers/Center '+neuron_name_list[use_neuron_index]+'(D,NumCenters)='+str((D,NoCenters))+'.txt')
                    # Center = np.load('Example/Centers_train25k_5k_D'+str(D)+'_tau'+str(tau)+'.npy')

                    stim_train = Current_train
                    Pdata = Voltage_test
                    beta_arr,R_arr = [10e-8,10e-4],[10e-5,10e-4,10e-3]#[10e-9,10e-8,10e-7,10e-6,10e-5,10e-4,10e-3,10e-2,10e-1]#[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2],[10e-4,10e-3,10e-2,10e-1,10e0,10e1,10e2]#,[10e-3,10e-2,10e-1,10e0,10e1,10e2,10e3]
                    bias = 150 # should be larger than tau*(D-1) or something like that
                    X = np.arange(bias,bias+PreLength*TT,TT)
                    stim_test = Current_test


                    beta = 10e-2
                    R = 10e-8
                    title = neuron_name_list[use_neuron_index]+' with 0.02 time step, D = '+str(D)+', Beta = '+str("{:.1e}".format(beta))+', R = '+str("{:.1e}".format(R))+' Train Time = '+str(length)+', Centers = '+str(NoCenters)+', tau = '+str(tau)+', lpf tau ='+str(fc_tau)+'cutoff frequency ='+str(1 / (2 * np.pi * fc_tau))+', transition band ='+str(b)
                    print(R)
                    F = DDF.FuncApproxF(Xdata,length,Center,beta,R,D,stim_train,tau)
                    try:
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
                        plt.savefig("Output/"+title+'.png')
                        # plt.show
                        plt.close()

                        '''
                        plt.plot(Voltage_train[:100000])
                        plt.plot(Pdata[0:180000])
                        plt.show()
                        plt.figure()
                        plt.plot(np.sort(DDF.W))
                        plt.title("RBF Coefficients (Sorted)")
                        plt.xlabel('Index (Sorted)')
                        plt.ylabel("RBF Coefficient Value")
                        plt.savefig("Output/"+title+'RBF Coefficients (Sorted).png')
                        plt.show()
                        
                        plt.figure()
                        plt.plot(np.sort(Center[:, 0]))
                        plt.title("Centers")
                        plt.xlabel("Sorted centers index")
                        plt.ylabel("Voltage")
                        plt.savefig("Output/"+title+'Centers.png')
                        plt.show()
                        
                
                
                        plt.scatter(Center[:, 0], DDF.W[:-1])
                        plt.title("Weight as a function of center voltage (unsure)")
                        plt.xlabel("Center voltage")
                        plt.ylabel("Weight (Coeff of RBF)")
                        plt.savefig("Output/"+title+'WeightsVSCenterVoltage.png')
                        plt.show()
                        '''

                    except:
                        print("PredValidation Error\n")




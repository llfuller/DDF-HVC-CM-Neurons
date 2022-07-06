#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TimeDelay_Neuron_DDF_GaussianForm import *
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import copy
from sklearn import preprocessing
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

neuron_name_list = ['Neuron_52_6-8-16(HVC_RA)',
                    'Neuron_57_6-8-16(HVC_x)',
                    'Neuron_61_6-8-16(HVC_I)',
                    'Neuron_8-15-2019']
directories_list = ['HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 52/',
                   'HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 57/',
                   'HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 61/',
                   'HVC_ra_x_i_data_2016_2019/08-15-2019/']

#neuron 2019


i = 0;
#FFT

# use_neuron_index = 3

fc = 0.009
b = 0.001
N = int(np.ceil((4 / b)))

if not N % 2: N += 1
n = np.arange(N)

sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
sinc_func = sinc_func * window
sinc_func = sinc_func / np.sum(sinc_func)

#neuron 52 from 2016 (HVC_RA)
use_neuron_index = i
# train 1, 14, test 16 (34 looks same as 1)
loaded_V_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_voltage.txt')
loaded_I_1 = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-0001_current.txt')

Voltage_train = loaded_V_1[:60000]
Current_train = np.convolve(loaded_I_1, sinc_func, "same")[:60000]
Voltage_test = loaded_V_1[60000:120000]
Current_test = np.convolve(loaded_I_1, sinc_func, "same")[60000:120000]
length = Voltage_train.shape[0]-1000
PreLength = Voltage_test.shape[0]-1000
print(Voltage_test.shape)
#neuron 57 from 2016 (HVC_x)
# use_neuron_index = 1

# still neuron 57 from 2016 (HVC-x)


#neuron 61 from 2016 (HVC_I)
# use_neuron_index = 2



new_signal = Current_train

print("Here 1")
print(np.shape(Voltage_train))
print(np.shape(Current_train))
print(np.shape(Current_test))
# Voltage_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00014_voltage.txt')
# Current_train = np.loadtxt(directories_list[use_neuron_index]+'nidaq_Dev1_ai-00014_current.txt')
print("Here 2")
print(np.shape(Voltage_train))
print(np.shape(Current_train))

sampling_rate = 30.0

fourier_transform = np.fft.rfft(Voltage_train)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
maxfreq = np.max(np.abs(power_spectrum))
norm_power_spectrum = power_spectrum / maxfreq
plt.figure()
plt.plot(frequency, norm_power_spectrum)
plt.title("Power freq of voltage ("+str(neuron_name_list[i])+")")
plt.xlabel('frequency',fontsize=20)
plt.ylabel('Voltage Power (normalized)',fontsize=20)
plt.xlim((0,1))
plt.legend()





#plt.savefig('Validation Prediction Result')
plt.savefig("Output/VoltageValidationNeuron"+str(neuron_name_list[i])+'ZoomOut.png')

with open('Output/VoltageValidationNeuron'+str(neuron_name_list[i])+'ZoomOut.txt', 'w') as f:
    f.write("Frequency \t\t\t Normalized Voltage Power Spectrum \n")
    for j in range(0, 2000):
        f.write(str(frequency[j])+'\t\t\t\t'+str(norm_power_spectrum[j])+'\n')
        

fourier_transform = np.fft.rfft(new_signal)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
maxfreq = np.max(np.abs(power_spectrum))
norm_power_spectrum = power_spectrum / maxfreq
plt.figure()
plt.plot(frequency, power_spectrum)
plt.title("Power freq of Current (Neuron"+str(neuron_name_list[i])+")")
plt.xlabel('frequency',fontsize=20)
plt.ylabel('Current Power (normalized)',fontsize=20)
plt.xlim((0,1))
plt.legend()

#plt.savefig('Validation Prediction Result')
plt.savefig("Output/CurrentValidationNeuron"+str(neuron_name_list[i])+'ZoomOut.png')

with open('Output/CurrentValidationNeuron'+str(neuron_name_list[i])+'ZoomOut.txt', 'w') as f:
    f.write("Frequency \t\t\t Current Power Spectrum \n")
    for j in range(0, 2000):
        f.write(str(frequency[j])+'\t\t\t\t'+str(power_spectrum[j])+'\n')

fourier_transform = np.fft.rfft(Voltage_train)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
maxfreq = np.max(np.abs(power_spectrum[10:]))
norm_power_spectrum = power_spectrum / maxfreq
plt.figure()
plt.plot(frequency[10:], norm_power_spectrum[10:])
plt.title("Power freq of voltage ("+str(neuron_name_list[i])+")")
plt.xlabel('frequency',fontsize=20)
plt.ylabel('Voltage Power (normalized)',fontsize=20)
plt.xlim((0,0.2))
plt.legend()

#plt.savefig('Validation Prediction Result')
plt.savefig("Output/VoltageValidationNeuron"+str(neuron_name_list[i])+'.png')

with open('Output/VoltageValidationNeuron'+str(neuron_name_list[i])+'.txt', 'w') as f:
    f.write("Frequency \t\t\t Normalized Voltage Power Spectrum \n")
    for j in range(10, 400):
        f.write(str(frequency[j])+'\t\t\t\t'+str(norm_power_spectrum[j])+'\n')

fourier_transform = np.fft.rfft(new_signal)
abs_fourier_transform = np.abs(fourier_transform)
power_spectrum = np.square(abs_fourier_transform)
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
maxfreq = np.max(np.abs(power_spectrum))
norm_power_spectrum = power_spectrum / maxfreq
plt.figure()
plt.plot(frequency, power_spectrum)
plt.title("Power freq of Current (Neuron"+str(neuron_name_list[i])+")")
plt.xlabel('frequency',fontsize=20)
plt.ylabel('Current Power (normalized)',fontsize=20)
plt.xlim((0,0.2))
plt.legend()

#plt.savefig('Validation Prediction Result')
plt.savefig("Output/CurrentValidationNeuron"+str(neuron_name_list[i])+'.png')

with open('Output/CurrentValidationNeuron'+str(neuron_name_list[i])+'.txt', 'w') as f:
    f.write("Frequency \t\t\t Current Power Spectrum \n")
    for j in range(10, 400):
        f.write(str(frequency[j])+'\t\t\t\t'+str(power_spectrum[j])+'\n')


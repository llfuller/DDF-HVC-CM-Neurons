import numpy as np
import matplotlib.pyplot as plt
import externally_provided_currents


def Fourier_Power_Spectrum_plot_and_save(data, name, sampling_rate, mean, xlim=175):
    # Training Current with no modifications
    # print("Plotting Fourier Power Spectrum for "+str(name))
    fourier_transform = np.fft.rfft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    # print("Length of power_spectrum is " + str(np.shape(power_spectrum)))
    frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))
    # print("Number of frequencies plotted is " + str(np.shape(frequency)))
    delta_freq = frequency[3] - frequency[2]
    # print("Frequency spacing is " + str(delta_freq))

    plt.figure()
    freq_without_0_index = frequency[1:]
    normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))

    plt.plot(freq_without_0_index,
             normalized_power_spec_without_0_index / np.max(np.abs(normalized_power_spec_without_0_index)))
    plt.title("Power(freq) of training current from ("+str(name)+")")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Power (1.0= max from spectrum times [1:])")
    plt.xlim(0, 175)
    plt.savefig("range="+str(mean)+"pA/Power spectrum of training current from ("+str(name)+").png")
    # plt.show()
    np.savetxt("range="+str(mean)+"pA/Power spectrum of training current from ("+str(name)+") range="+str(mean)+".txt",
               np.column_stack((freq_without_0_index, normalized_power_spec_without_0_index)))

def Data_plot_and_save(data, t_arr, mean, name):
    plt.figure()
    plt.plot(t_arr, data)
    plt.title('Current vs Time')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Current (pA)")
    plt.savefig("range="+str(mean)+"pA/Current vs Time ("+str(name)+").png")
    # plt.show()
    # np.savetxt("max="+str(mean)+"pA/"+name+"max="+str(max)+"_(t,I).txt",
    #            np.concatenate((t_arr[:, np.newaxis], data[:, np.newaxis]), axis=1))
    np.savetxt("range="+str(mean)+"pA/"+name+"range="+str(mean)+"_(I).txt",
               np.concatenate((t_arr[:, np.newaxis], data[:, np.newaxis]), axis=1))

def set_mean(arr, new_mean=1):
    # return np.divide(arr,np.mean(arr))*new_mean
    # return np.divide(arr,np.max(arr))*new_mean
    return None
def set_range(arr, lower_bound, upper_bound):
    # print("Upper-lower="+str(upper_bound - lower_bound))
    # print(arr+np.min(arr))
    # print(np.max(arr)-np.min(arr))
    # print(lower_bound)
    return np.divide(arr+np.abs(np.min(arr)),np.max(arr)+np.abs(np.min(arr)))*(upper_bound + np.abs(lower_bound)) - np.abs(lower_bound)


# for mean_value in [200,300]:
for dilation_factor in [0.5,1.0,3.0]: # dilation_factor of 1 = good and normal Fourier Power Spectrum.
    print("Dilation factor:"+str(dilation_factor))
    for max_value in [100,200,300,400,500,600,700]:

        # --------------- Current used on Meliza Data --------------
        mean_value=max_value
        # Current vs Time
        print("Initial load")
        VIT = np.loadtxt("2014_12_11_0017_VIt.txt")
        t_arr = VIT[:,2]
        I_arr = VIT[:,1]

        # I_2014_12_11_0017 = set_mean(I_arr, new_mean=mean_value)
        I_2014_12_11_0017 = set_range(I_arr, -float(max_value)/3.0, max_value)

        Data_plot_and_save(I_2014_12_11_0017, t_arr, mean=mean_value, name= "I_920061fe_2014_12_11_0017")

        # timestep = float(t_arr[3]-t_arr[2])
        timestep = 0.00002 #seconds
        sampling_rate = 1.0/(timestep)
        # print("Timestep: "+str(timestep) +str("(seconds)"))
        # print("Timestep: "+str(1000*(timestep)) +str("(ms)"))
        # print("Sampling rate: "+str(sampling_rate)+str("(Hz)"))

        # print("Reloading to make sure it works")
        # VIT = np.loadtxt("I_920061fe_2014_12_11_0017_(t,I).txt")
        # t_arr = VIT[:,0]
        # I_arr = VIT[:,1]
        plt.figure()
        plt.plot(t_arr,I_2014_12_11_0017)
        plt.title('Reloaded Current vs Time')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Current (pA)")
        # plt.show()
        # print("Timestep: "+str(t_arr[3]-t_arr[2]) +str("(seconds)"))
        # print("Timestep: "+str(1000*(t_arr[3]-t_arr[2])) +str("(ms)"))
        # print("Sampling rate: "+str(1.0/(t_arr[3]-t_arr[2]))+str("(Hz)"))
        Fourier_Power_Spectrum_plot_and_save(data=I_2014_12_11_0017, name="I_920061fe_2014_12_11_0017", mean=mean_value, sampling_rate=sampling_rate)

        # Data_plot_and_save(I_arr, t_arr, "I_920061fe_2014_12_11_0017")




        #====================================================================================================
        # Other currents
        #==================================================================================================
        # mean_value = np.mean(I_arr)

        timestep = 0.00002 #seconds
        sampling_rate = 1.0/(timestep)
        total_time = 15 #seconds
        t_arr = np.arange(start=0,stop=total_time,step=timestep)
        # print("Number of indices in t_arr:"+str(t_arr.shape[0]))
        # print("Sampling rate: "+str(sampling_rate)+str("(Hz)"))


        # --------------- L63 --------------
        plot_3D = True
        scaling_time = dilation_factor*22.0#200
        L63_obj = externally_provided_currents.L63_object(scaling_time_factor=scaling_time)
        I_L63_x, I_L63_y, I_L63_z = L63_obj.prepare_f(t_arr).T
        I_L63_x = set_range(I_L63_x, -float(max_value)/3.0, max_value)
        I_L63_y = set_range(I_L63_y, -float(max_value)/3.0, max_value)
        I_L63_z = set_range(I_L63_z, -float(max_value)/3.0, max_value)

        # I_L63_x = set_mean(I_L63_x, new_mean=mean_value)
        # I_L63_y = set_mean(I_L63_y, new_mean=mean_value)
        # I_L63_z = set_mean(I_L63_z, new_mean=mean_value)


        Data_plot_and_save(I_L63_x, t_arr, mean=mean_value, name="I_L63_x_time_dilation="+str(dilation_factor))

        if plot_3D == True:
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(I_L63_x[0:-1:10], I_L63_y[0:-1:10], I_L63_z[0:-1:10], c=t_arr[0:-1:10], cmap='winter')
            # plt.show()

        Data_plot_and_save(I_L63_x, t_arr, mean=mean_value, name= "I_L63_x_time_dilation="+str(dilation_factor))
        Data_plot_and_save(I_L63_y, t_arr, mean=mean_value, name= "I_L63_y_time_dilation="+str(dilation_factor))
        Data_plot_and_save(I_L63_z, t_arr, mean=mean_value, name= "I_L63_z_time_dilation="+str(dilation_factor))

        Fourier_Power_Spectrum_plot_and_save(data=I_L63_x, name="I_L63_x_time_dilation="+str(dilation_factor), mean=mean_value, sampling_rate=sampling_rate)

        # --------------- Colpitts --------------
        plot_3D = True
        scaling_time = dilation_factor*150.0#200
        colp_obj = externally_provided_currents.Colpitts_object(scaling_time_factor=scaling_time)

        print(colp_obj.prepare_f(t_arr).shape)
        I_colpitts_x, I_colpitts_y, I_colpitts_z = colp_obj.prepare_f(t_arr).T
        # if plot_3D == True:
        #     plt.figure()
        #     ax = plt.axes(projection='3d')
        #     ax.scatter3D(I_colpitts_x[0:-1:10], I_colpitts_y[0:-1:10], I_colpitts_z[0:-1:10], c=t_arr[0:-1:10], cmap='winter')
        #     plt.show()
        I_colpitts_x = set_range(I_colpitts_x, -float(max_value)/3.0, max_value)
        I_colpitts_y = set_range(I_colpitts_y, -float(max_value)/3.0, max_value)
        I_colpitts_z = set_range(I_colpitts_z, -float(max_value)/3.0, max_value)

        # I_colpitts_x = set_mean(I_colpitts_x, new_mean=mean_value)
        # I_colpitts_y = set_mean(I_colpitts_y, new_mean=mean_value)
        # I_colpitts_z = set_mean(I_colpitts_z, new_mean=mean_value)

        Data_plot_and_save(I_colpitts_x, t_arr, mean=mean_value, name= "I_colpitts_x_time_dilation="+str(dilation_factor))
        Fourier_Power_Spectrum_plot_and_save(data=I_colpitts_x, name="I_colpitts_x_time_dilation="+str(dilation_factor), mean=mean_value, sampling_rate=sampling_rate)
        Data_plot_and_save(I_colpitts_y, t_arr, mean=mean_value, name= "I_colpitts_y_time_dilation="+str(dilation_factor))
        Fourier_Power_Spectrum_plot_and_save(data=I_colpitts_y, name="I_colpitts_y_time_dilation="+str(dilation_factor), mean=mean_value, sampling_rate=sampling_rate)
        Data_plot_and_save(I_colpitts_z, t_arr, mean=mean_value, name= "I_colpitts_z_time_dilation="+str(dilation_factor))
        Fourier_Power_Spectrum_plot_and_save(data=I_colpitts_z, name="I_colpitts_z_time_dilation="+str(dilation_factor), mean=mean_value, sampling_rate=sampling_rate)


        # --------------- L96 --------------
        scaling_time = dilation_factor*22.0
        L96_obj = externally_provided_currents.L96_object(scaling_time_factor=scaling_time)
        # L96_obj.alpha = scaling_time*L96_obj.alpha

        print(L96_obj.prepare_f(t_arr).shape)
        L96_x1 = (L96_obj.prepare_f(t_arr).T)[0]
        L96_x1[:4000] = 0 # remove transients
        # L96_x1 = set_mean(L96_x1, new_mean=mean_value)
        L96_x1 = set_range(L96_x1, -float(max_value)/3.0, max_value)

        Data_plot_and_save(L96_x1, t_arr, mean=mean_value, name= "I_L96_x1_time_dilation="+str(dilation_factor))
        Fourier_Power_Spectrum_plot_and_save(data=L96_x1, name="I_L96_x1_time_dilation="+str(dilation_factor), mean=mean_value, sampling_rate=sampling_rate)

    # for max_value in [100, 200, 300, 400, 500, 600, 700]:
    #
    #     # --------------- Current used on Meliza Data --------------
    #     mean_value = max_value
    #     # Current vs Time
    #     print("Initial load")
    #     VIT = np.loadtxt("2014_12_11_0017_VIt.txt")
    #     t_arr = VIT[:, 2]
    #     I_arr = VIT[:, 1]
    #
    #     # I_2014_12_11_0017 = set_mean(I_arr, new_mean=mean_value)
    #     I_2014_12_11_0017 = set_range(I_arr, -float(max_value) / 3.0, max_value)
    #
    #     Data_plot_and_save(I_2014_12_11_0017, t_arr, mean=mean_value, name="I_920061fe_2014_12_11_0017")
    #
    #     # timestep = float(t_arr[3]-t_arr[2])
    #     timestep = 0.00002  # seconds
    #     sampling_rate = 1.0 / (timestep)
    #     print("Timestep: " + str(timestep) + str("(seconds)"))
    #     print("Timestep: " + str(1000 * (timestep)) + str("(ms)"))
    #     print("Sampling rate: " + str(sampling_rate) + str("(Hz)"))
    #
    #     # print("Reloading to make sure it works")
    #     # VIT = np.loadtxt("I_920061fe_2014_12_11_0017_(t,I).txt")
    #     # t_arr = VIT[:,0]
    #     # I_arr = VIT[:,1]
    #     plt.figure()
    #     plt.plot(t_arr, I_2014_12_11_0017)
    #     plt.title('Reloaded Current vs Time')
    #     plt.xlabel("Time (seconds)")
    #     plt.ylabel("Current (pA)")
    #     plt.show()
    #     print("Timestep: " + str(t_arr[3] - t_arr[2]) + str("(seconds)"))
    #     print("Timestep: " + str(1000 * (t_arr[3] - t_arr[2])) + str("(ms)"))
    #     print("Sampling rate: " + str(1.0 / (t_arr[3] - t_arr[2])) + str("(Hz)"))
    #     Fourier_Power_Spectrum_plot_and_save(data=I_2014_12_11_0017, name="I_920061fe_2014_12_11_0017",
    #                                          mean=mean_value, sampling_rate=sampling_rate)
    #
    #     # Data_plot_and_save(I_arr, t_arr, "I_920061fe_2014_12_11_0017")
    #
    #     # ====================================================================================================
    #     # Other currents
    #     # ==================================================================================================
    #     # mean_value = np.mean(I_arr)
    #
    #     timestep = 0.00002  # seconds
    #     sampling_rate = 1.0 / (timestep)
    #     total_time = 15  # seconds
    #     t_arr = np.arange(start=0, stop=total_time, step=timestep)
    #     print("Number of indices in t_arr:" + str(t_arr.shape[0]))
    #     print("Sampling rate: " + str(sampling_rate) + str("(Hz)"))
    #
    #     # --------------- L63 --------------
    #     plot_3D = True
    #     scaling_time = dilation_factor * 22.0  # 200
    #     L63_obj = externally_provided_currents.L63_object(scaling_time_factor=scaling_time)
    #     I_L63_x, I_L63_y, I_L63_z = L63_obj.prepare_f(t_arr).T
    #     I_L63_x = set_range(I_L63_x, -float(max_value) / 3.0, max_value)
    #     I_L63_y = set_range(I_L63_y, -float(max_value) / 3.0, max_value)
    #     I_L63_z = set_range(I_L63_z, -float(max_value) / 3.0, max_value)
    #
    #     # I_L63_x = set_mean(I_L63_x, new_mean=mean_value)
    #     # I_L63_y = set_mean(I_L63_y, new_mean=mean_value)
    #     # I_L63_z = set_mean(I_L63_z, new_mean=mean_value)
    #
    #     Data_plot_and_save(I_L63_x, t_arr, mean=mean_value, name="I_L63_x")
    #
    #     if plot_3D == True:
    #         plt.figure()
    #         ax = plt.axes(projection='3d')
    #         ax.scatter3D(I_L63_x[0:-1:10], I_L63_y[0:-1:10], I_L63_z[0:-1:10], c=t_arr[0:-1:10], cmap='winter')
    #         plt.show()
    #
    #     Data_plot_and_save(I_L63_x, t_arr, mean=mean_value, name="I_L63_x")
    #     Data_plot_and_save(I_L63_y, t_arr, mean=mean_value, name="I_L63_y")
    #     Data_plot_and_save(I_L63_z, t_arr, mean=mean_value, name="I_L63_z")
    #
    #     Fourier_Power_Spectrum_plot_and_save(data=I_L63_x, name="I_L63_x", mean=mean_value,
    #                                          sampling_rate=sampling_rate)
    #
    #     # --------------- Colpitts --------------
    #     plot_3D = True
    #     scaling_time = dilation_factor * 150.0  # 200
    #     colp_obj = externally_provided_currents.Colpitts_object(scaling_time_factor=scaling_time)
    #
    #     print(colp_obj.prepare_f(t_arr).shape)
    #     I_colpitts_x, I_colpitts_y, I_colpitts_z = colp_obj.prepare_f(t_arr).T
    #     # if plot_3D == True:
    #     #     plt.figure()
    #     #     ax = plt.axes(projection='3d')
    #     #     ax.scatter3D(I_colpitts_x[0:-1:10], I_colpitts_y[0:-1:10], I_colpitts_z[0:-1:10], c=t_arr[0:-1:10], cmap='winter')
    #     #     plt.show()
    #     I_colpitts_x = set_range(I_colpitts_x, -float(max_value) / 3.0, max_value)
    #     I_colpitts_y = set_range(I_colpitts_y, -float(max_value) / 3.0, max_value)
    #     I_colpitts_z = set_range(I_colpitts_z, -float(max_value) / 3.0, max_value)
    #
    #     # I_colpitts_x = set_mean(I_colpitts_x, new_mean=mean_value)
    #     # I_colpitts_y = set_mean(I_colpitts_y, new_mean=mean_value)
    #     # I_colpitts_z = set_mean(I_colpitts_z, new_mean=mean_value)
    #
    #     Data_plot_and_save(I_colpitts_x, t_arr, mean=mean_value, name="I_colpitts_x")
    #     Fourier_Power_Spectrum_plot_and_save(data=I_colpitts_x, name="I_colpitts_x", mean=mean_value,
    #                                          sampling_rate=sampling_rate)
    #     Data_plot_and_save(I_colpitts_y, t_arr, mean=mean_value, name="I_colpitts_y")
    #     Fourier_Power_Spectrum_plot_and_save(data=I_colpitts_y, name="I_colpitts_y", mean=mean_value,
    #                                          sampling_rate=sampling_rate)
    #     Data_plot_and_save(I_colpitts_z, t_arr, mean=mean_value, name="I_colpitts_z")
    #     Fourier_Power_Spectrum_plot_and_save(data=I_colpitts_z, name="I_colpitts_z", mean=mean_value,
    #                                          sampling_rate=sampling_rate)
    #
    #     # --------------- L96 --------------
    #     scaling_time = dilation_factor * 22.0
    #     L96_obj = externally_provided_currents.L96_object(scaling_time_factor=scaling_time)
    #     # L96_obj.alpha = scaling_time*L96_obj.alpha
    #
    #     print(L96_obj.prepare_f(t_arr).shape)
    #     L96_x1 = (L96_obj.prepare_f(t_arr).T)[0]
    #     L96_x1[:4000] = 0  # remove transients
    #     # L96_x1 = set_mean(L96_x1, new_mean=mean_value)
    #     L96_x1 = set_range(L96_x1, -float(max_value) / 3.0, max_value)
    #
    #     Data_plot_and_save(L96_x1, t_arr, mean=mean_value, name="I_L96_x1")
    #     Fourier_Power_Spectrum_plot_and_save(data=L96_x1, name="I_L96_x1", mean=mean_value,
    #                                          sampling_rate=sampling_rate)
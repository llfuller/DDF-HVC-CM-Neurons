"""
Calculate FNN statistics
Written by Barry Xue and lightly edited with Lawson Fuller
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import time
import torch
import os
import re

print(torch.cuda.is_available())
print(f"torch.cuda.device_count(){torch.cuda.device_count()}\n")
print(f"torch.cuda.current_device(){torch.cuda.current_device()}\n")
print(f"torch.cuda.device(0){torch.cuda.device(0)}\n")
print(f"torch.cuda.get_device_name(0){torch.cuda.get_device_name(0)}\n")

"""Define all the functions"""
def count_fnn(dataset, threshold_R=1e-1):
    """
    Parameters:
    dataset: A dataset should contain all the points and their closest point in pairs;
    example [(point_vec1, point_vec2)_1, (point_vec1, point_vec2)_2, ...]

    threshold_R: This threshold determines ratio needed between the actual distance and the time delay distance to be
    recognized as a true nearest neighbor

    Return:
    A floating point number indicating the number of false nearest neighbors in the dataset.
    """
    tnn, fnn = 0, 0
    for vec1, vec2 in dataset:
        true1, true2 = vec1[0], vec2[0]# the first value of each array are the true voltage value
        time_dist = np.linalg.norm(vec1 - vec2, ord=2) # time delay distance between the two points
        actual_dist = np.abs(true1 - true2) # actual distance
        dist_ratio = actual_dist / time_dist

        if dist_ratio <= threshold_R: # determine falsehood
            tnn += 1
        else:
            fnn += 1
    return fnn / (fnn+tnn)


def generate_min_dist_datapoints(data, window=1000, save_data=True):
    result_data, result_index = [], []
    print(f"Received data with shape{data.shape}")

    for i in tqdm.tqdm(range(len(data))):

        # create a window of data point to search over
        # save the starting index and the ending index of our window as reference
        start_index, end_index = i-window, i+window#max(i-window, 0), min(len(data), i+window)

        # exclude the target point from the window
        search_window = torch.cat([
            data[(start_index<0)*start_index:0],
            data[(start_index<0)*0+(start_index>=0)*start_index:i],
            data[i+1:end_index*(end_index<data.shape[0]) + data.shape[0]*(end_index>=data.shape[0])],
            data[0:(end_index-data.shape[0])*(end_index>=data.shape[0])]
        ])
        # print(search_window)

        # run the distance calculation and find the closest point and their indices
        # print(f"data[i] is {data[i].shape}")
        distance = torch.norm(search_window - data[i], dim=1)
        # print(search_window - data[i])
        # print(torch.norm(search_window - data[i], dim=1))
        min_distance_index = torch.argmin(distance) # index of minimum distance point inside the window of datapoints
        # print(min_distance_index)
        # print(min_distance_index)
        min_distance_pair_data = [(data[i].data).cpu(), (search_window[min_distance_index].data).cpu()]

        # find the real index in respect to the entire dataset
        real_min_distance_index = None#start_index + min_distance_index + 1 if start_index + min_distance_index >= i else start_index + min_distance_index
        if start_index + min_distance_index < 0:
            real_min_distance_index = data.shape[0] + (start_index + min_distance_index)
        elif (start_index + min_distance_index) >= 0:
            if (start_index + min_distance_index) < i:
                real_min_distance_index = start_index + min_distance_index
            if (start_index + min_distance_index) >= i:
                if (start_index + min_distance_index) + 1 < data.shape[0]:
                    real_min_distance_index = start_index + min_distance_index + 1
                if (start_index + min_distance_index) + 1 >= data.shape[0]:
                    real_min_distance_index = start_index + min_distance_index + 1 - data.shape[0]
        else:
            print(f"Failed to match any case: (start_index + min_distance_index)={(start_index + min_distance_index)} and i={i}")
        min_distance_pair_index = [i, (real_min_distance_index.data).cpu()]

        # save the closest point's index
        # this is the k and j: index of the first and second data points
        result_data.append(min_distance_pair_data)
        result_index.append(min_distance_pair_index)

        # early stopping for time-saving purposes
        if i == int(round(len(data)/10)):
            break

    # for i in tqdm.tqdm(range(len(data))):
    #     result_data.append(min_distance_pair_data)
    #     result_index.append(min_distance_pair_index)


    # convert the list to numpy array
    result_data = np.array(result_data)
    result_index = np.array(result_index)

    # Functionality moved into MultiFile_FNN.py.
    # if save_data:
    #     print(f"Result data has total shape {result_data.shape}")
    #     print(f"Saving results for D={data.shape[1]}, window={window}.")
    #     np.save(f'data_derived/npy/min_datapairs_D={data.shape[1]}_window={window}_datapoints', result_data)
    #     np.save(f'data_derived/npy/min_datapairs_D={data.shape[1]}_window={window}_location', result_index)
    #     print("Results saved.")
    return result_data, result_index


def generate_min_dist_datasets(filepath, time_delay_datasets, tau=5, R_ratio=1e-2, D_arr=np.array([1, 2, 4, 6, 8, 10, 12, 15, 18, 20]), search_window_size=1000, save_data=True):

    """Search for minimum distance pairs
        # Returns: list with # D_E tried elements, each element a thing (datatructure varies) with first dimension length 2
        # second dimension of the thing is dimension D_E for datasets, and 2 for time_delay_indices or time_delay_datapairs
    """
    # change the data from array to tensor for faster calculation
    device = torch.device('cuda:0') # change device index
    time_delay_datasets = [(torch.tensor(arr)).to(device) for arr in time_delay_datasets]

    # calculate the min distance result (test) - window=200 will run for 20 mins
    time_delay_datapairs, time_delay_indices = [], []
    for time_delay_data in time_delay_datasets:
        print(f"Iterating for time delay data with shape {time_delay_data.shape}")
        data, index = generate_min_dist_datapoints(time_delay_data, search_window_size, save_data)
        time_delay_datapairs.append(data)
        time_delay_indices.append(index)


    return time_delay_datasets, time_delay_indices, time_delay_datapairs


def count_fnn(index, dataset, threshold_R=1e-2):
    """
    Parameters:
    dataset: A dataset should contain all the points and their closest point in pairs;
    example [(point_vec1, point_vec2)_1, (point_vec1, point_vec2)_2, ...]

    threshold_R: This threshold determines ratio needed between the actual distance and
    the time delay distance to be recognized as a true nearest neighbor

    Return:
    A floating point number indicating the number of false nearest neighbors in the dataset.
    """
    counter = 0
    tnn, fnn = 0, 0
    for i in range(index.shape[0]):
        td1, td2 = dataset[i]  # time delay vectors
        true1, true2 = td1[0], td2[0]  # the first value of each time delay vectors are the true voltage value
        time_dist = np.linalg.norm(td1 - td2, ord=2)  # time delay distance between the two points
        actual_dist = np.abs(true1 - true2)  # actual distance
        dist_ratio = actual_dist / time_dist

        # dist_ratio = actual_dist / time_dist if time_dist != 0 else 0
        if dist_ratio <= threshold_R:  # determine falsehood
            tnn += 1
        else:
            fnn += 1
        counter += 1

    return fnn / (fnn + tnn)


def run_this(loaded_I, loaded_V, loaded_t,
             filepath = "Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/epoch_1.txt",
             search_window_size = 100000,
             tau = 5,
             R_ratio = 1e-2,
             D_arr = np.array([1, 2, 4, 6, 8, 10, 12, 15, 18, 20]),
             save_data = True):
    """User Defined Parameters: specify filepath, hyperparameters
        R_ratio: distance ratio threshold R
    """
    # get target data (Lilac 114, Neuron 1, epoch_1.txt)
    # define tau (user provided), distance ratio threshold R (user defined hyperparameter), and range of D to search
    # over (trainable parameter)

    # Functionality of the below commented section moved into MultiFile_FNN.py, but kept here for reference
    # Import data using filepath
    # # Column1: Current, Column2: Voltage
    # try:
    #     lilac_114_1_1 = pd.read_csv(filepath, delimiter='\t', header=None)
    # except:
    #     lilac_114_1_1 = pd.read_csv(filepath, delimiter='\t', header=None)
    # lilac_114_1_1.reset_index(inplace=True)
    # lilac_114_1_1.columns = ['Time', 'Current', 'Voltage']
    # lilac_114_1_1.head()
    #
    # # Change stored types of data
    # original_data = lilac_114_1_1.to_numpy()
    # T = original_data[:, 0].astype(np.int64)
    # I = original_data[:, 1].astype(np.float64)
    # V_0 = original_data[:, 2].astype(np.float64).reshape((len(T), 1)) # voltage at 0*tau

    # units for I, V, T, likely same as in Multifile: (probably pA, mV, ms)
    T = loaded_t.astype(np.int64)
    I = loaded_I.astype(np.float64)
    V_0 = loaded_V.astype(np.float64).reshape((len(T), 1))
    # create the time delay vectors for each data points with each of the D values
    start = time.time()
    time_delay_datasets = []
    for d in D_arr:
        V_s = np.dstack([np.concatenate([V_0[-i*tau:, :], V_0[:-i*tau, :]], axis=0) for i in range(d+1)])[:, 0, :]
        time_delay_datasets.append(V_s)
        # print(f"Length of time delay data appended: {V_s.shape}")
    end = time.time()
    print(f"This took {end-start}.")


    """Run Min-distance pair search"""
    # returning list with # D_E tried elements, each element a thing (datatructure varies) with first dimension length 2
    # second dimension of the thing is dimension D_E for datasets, and 2 for time_delay_indices or time_delay_datapairs
    datasets, time_delay_indices, time_delay_datapairs = generate_min_dist_datasets(filepath,time_delay_datasets,
                                                                                    tau, R_ratio, D_arr,
                                                                                    search_window_size, save_data)


    """
    TODO: Run the FNN algorithm here
    - multifile_fps doesn't generate vector for different D_E values, gotta do it for fnn again
    """
    # convert returned list vectors to numpy
    datasets_numpy = []
    for a_dataset_tensor_on_gpu in datasets:
        datasets_numpy.append(a_dataset_tensor_on_gpu.cpu().numpy())

    return datasets_numpy, time_delay_indices, time_delay_datapairs
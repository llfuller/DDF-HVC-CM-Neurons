"""
Calculate FNN statistics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import time
import torch
import os
import re


"""Define all the functions"""
def count_fnn(dataset, threshold_R=1e-1):
    """
    Parameters:
    dataset: A dataset should contain all the points and their closest point in pairs; example [(point_vec1, point_vec2)_1, (point_vec1, point_vec2)_2, ...]
    threshold_R: This threshold determines ratio needed between the actual distance and the time delay distance to be recognized as a true nearest neighbor

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
    for i in tqdm(range(len(data))):

        # create a window of data point to search over
        # save the starting index and the ending index of our window as reference
        start_index, end_index = max(i-window, 0), min(len(data), i+window)

        # exclude the target point from the window
        search_window = torch.cat([
            data[start_index:i],
            data[i+1:end_index]
        ])
        # print(search_window)

        # run the distance calculation and find the closest point and their indices
        distance = torch.norm(search_window - data[i], dim=1)
        # print(search_window - data[i])
        # print(torch.norm(search_window - data[i], dim=1))
        min_distance_index = torch.argmin(distance) # index of minimum distance point inside the window of datapoints
        print(min_distance_index)
        min_distance_pair_data = [data[i].data, search_window[min_distance_index].data]

        # find the real index in respect to the entire dataset
        real_min_distance_index = start_index + min_distance_index + 1 if start_index + min_distance_index >= i else start_index + min_distance_index
        min_distance_pair_index = [i, real_min_distance_index.data]

        # save the closest point's index
        # this is the k and j: index of the first and second data points
        result_data.append(min_distance_pair_data)
        result_index.append(min_distance_pair_index)

        # early stopping for testing purposes
        if i == 0:
            break

    # convert the list to numpy array
    result_data = np.array(result_data)
    result_index = np.array(result_index)

    if save_data:
        print("Saving results...")
        np.save(f'min_datapairs_D={result_data.shape[1]}_window={window}_datapoints', result_data)
        np.save(f'min_datapairs_D={result_index.shape[1]}_window={window}_location', result_index)
        print("Results saved.")
    return result_data, result_index


def generate_min_dist_datasets(filepath, tau=5, R_ratio=1e-2, D_arr=np.array([1, 2, 4, 6, 8, 10, 12, 15, 18, 20]), search_window_size=1000, save_data=True):
    # Column1: Current, Column2: Voltage
    try:
        lilac_114_1_1 = pd.read_csv(filepath, delimiter='\t', header=None)
    except:
        lilac_114_1_1 = pd.read_csv(filepath, delimiter='\t', header=None)
    lilac_114_1_1.reset_index(inplace=True)
    lilac_114_1_1.columns = ['Time', 'Current', 'Voltage']
    lilac_114_1_1.head()

    # create the time delay vectors for each data points with each of the D values
    original_data = lilac_114_1_1.to_numpy()
    T = original_data[:, 0].astype(np.int64)
    I = original_data[:, 1].astype(np.float64)
    V_0 = original_data[:, 2].astype(np.float64).reshape((len(T), 1)) # voltage at 0*tau


    start = time.time()
    time_delay_datasets = []
    for d in D_arr:
        V_s = np.dstack([np.concatenate([V_0[-i*tau:, :], V_0[:-i*tau, :]], axis=0) for i in range(d+1)])[:, 0, :]
        time_delay_datasets.append(V_s)
    end = time.time()
    print(f"This took {end-start}.")


    """Search for minimum distance pairs"""
    # change the data from array to tensor for faster calculation
    torch.device('cuda') # change device index
    time_delay_datasets = [torch.tensor(arr) for arr in time_delay_datasets]

    # calculate the min distance result (test) - window=200 will run for 20 mins
    time_delay_datapairs, time_delay_indices = [], []
    for time_delay_data in time_delay_datasets:
        data, index = generate_min_dist_datapoints(time_delay_data, search_window_size, save_data)
        time_delay_datapairs.append(data)
        time_delay_indices.append(index)

    return time_delay_datasets, time_delay_indices

"""User Defined Parameters: specify filepath, hyperparameters"""

# get target data (Lilac 114, Neuron 1, epoch_1.txt)
filepath = "./Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/epoch_1.txt"
search_window_size = 100000
save_data = True
# define tau (user provided), distance ratio threshold R (user defined hyperparameter), and range of D to search over (trainable parameter)
tau = 5
R_ratio = 1e-2
D_arr = np.array([1, 2, 4, 6, 8, 10, 12, 15, 18, 20])

"""Run Min-distance pair search"""
datasets, time_delay_indices = generate_min_dist_datasets(filepath, tau, R_ratio, D_arr, search_window_size, save_data)

"""User Defined Parameters: specify root path for loaded files"""
root = "./FNN/FNN_min-dist_results/"

"""Load Saved min distance datasets"""
# load from all the npy file
D_data_dict, D_index_dict = {}, {}

try:
    os.listdir(root)
except:
    root = "./FNN/"

for filepath in os.listdir(root):
    if 'min_datapairs' not in filepath:
        continue
    # print(filepath)
    D = int(re.search("D=([0-9]*)", filepath).group(1))
    window_size = int(re.search("window=([0-9]*)", filepath).group(1))
    data = True if re.search("datapoints", filepath) else False
    if data:
        D_data_dict[(D, window_size)] = np.load(root+filepath)
    else:
        D_index_dict[(D, window_size)] = np.load(root+filepath)

"""
TODO: Run the FNN algorithm here
- multifile_fps doesn't generate vector for different D_E values, gotta do it for fnn again
"""
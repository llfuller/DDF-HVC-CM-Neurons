"""
Run this script to get minimum distanced data pairs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch

def generate_min_dist_datapoints(data, device, window=100000, get_data=False):
    # fix window size if it is too big
    if window > len(data):
        window = len(data)-1

    result_data, result_index = torch.zeros((data.shape[0], 2, data.shape[1]), device=device), torch.zeros((data.shape[0], 2), device=device)
    for i in tqdm(range(len(data))):

        # create a window of data point to search over
        # save the starting index and the ending index of our window as reference
        start_index, end_index = max(i-window, 0), min(len(data), i+window)

        # exclude the target point from the window
        search_window = torch.cat([
            data[start_index:i],
            data[i+1:end_index]
        ])

        # run the distance calculation and find the closest point and their indices
        distance = torch.norm(search_window - data[i], dim=1)
        min_distance_index = torch.argmin(distance) # index of minimum distance point inside the window of datapoints
        result_data[i, 0, :] = data[i]
        result_data[i, 1, :] = search_window[min_distance_index]

        # find the real index in respect to the entire dataset
        real_min_distance_index = start_index + min_distance_index + 1 if start_index + min_distance_index >= i else start_index + min_distance_index
        result_index[i, 0] = i
        result_index[i, 1] = real_min_distance_index

        # save the closest point's index
        # this is the k and j: index of the first and second data points
        # result_data.append(min_distance_pair_data)
        # result_index.append(min_distance_pair_index)

        # early stopping for testing purposes
        # if i > 100:
        #     break

    # convert the list to numpy array
    # result_data = np.array(result_data)
    # result_index = np.array(result_index)


    print("Saving results...")
    np.save(f'min_datapairs_D={data.shape[1]-1}_window={window}_datapoints', result_data.cpu().data, allow_pickle=True)
    np.save(f'min_datapairs_D={data.shape[1]-1}_window={window}_location', result_index.cpu().data, allow_pickle=True)
    print("Results saved.")
    if get_data:
        return result_data, result_index

"""Here starts the actual executable script"""

"""
Specify file source, tau, R, D values before running the script
"""

# get target data (Lilac 114, Neuron 1, epoch_1.txt)
# Column1: Current, Column2: Voltage
try:
    lilac_114_1_1 = pd.read_csv('../Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/epoch_1.txt', delimiter='\t', header=None)
except:
    lilac_114_1_1 = pd.read_csv('../Data2022-50KhZ/Lilac 114/Neuron 1/epoch_1.txt', delimiter='\t', header=None)
lilac_114_1_1.reset_index(inplace=True)
lilac_114_1_1.columns = ['Time', 'Current', 'Voltage']
lilac_114_1_1.head()

# define tau (user provided), distance ratio threshold R (user defined hyperparameter), and range of D to search over (trainable parameter)
tau = 5
R_ratio = 10 # TODO: Investigate optimal value of R; 10 is suggested by the original paper
D_arr = np.array([1, 2, 4, 6, 8, 10, 12, 15, 18, 20])

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


# change the data from array to tensor for faster calculation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Pytorch is using {device}")
time_delay_datasets = [torch.tensor(arr, device=device) for arr in time_delay_datasets]

for dataset in time_delay_datasets:
    generate_min_dist_datapoints(data=dataset, device=device) # data would be saved as numpy files
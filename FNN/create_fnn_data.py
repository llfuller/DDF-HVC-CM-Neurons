"""
Run this script to get minimum distanced data pairs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import time
import torch

def generate_min_dist_datapoints(data, window=1000, get_data=False):
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

        # run the distance calculation and find the closest point and their indices
        distance = torch.norm(search_window - data[i], dim=1)
        min_distance_index = torch.argmin(distance) # index of minimum distance point inside the window of datapoints
        min_distance_pair_data = [data[i].data, search_window[min_distance_index].data]

        # find the real index in respect to the entire dataset
        real_min_distance_index = start_index + min_distance_index + 1 if start_index + min_distance_index >= i else start_index + min_distance_index
        min_distance_pair_index = [i, real_min_distance_index.data]

        # save the closest point's index
        # this is the k and j: index of the first and second data points
        result_data.append(min_distance_pair_data)
        result_index.append(min_distance_pair_index)

        # early stopping for testing purposes
        # if i > 100:
        #     break

    # convert the list to numpy array
    result_data = np.array(result_data)
    result_index = np.array(result_index)


    print("Saving results...")
    np.save(f'min_datapairs_D={data.shape[1]}_window={window}_datapoints', result_data)
    np.save(f'min_datapairs_D={data.shape[1]}_window={window}_location', result_index)
    print("Results saved.")
    if get_data:
        return result_data, result_index

"""Here starts the actual executable script"""
# get target data (Lilac 114, Neuron 1, epoch_1.txt)
# Column1: Current, Column2: Voltage
try:
    lilac_114_1_1 = pd.read_csv('./Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/epoch_1.txt', delimiter='\t', header=None)
except:
    lilac_114_1_1 = pd.read_csv('./Data2022-50KhZ/Lilac 114/Neuron 1/epoch_1.txt', delimiter='\t', header=None)
lilac_114_1_1.reset_index(inplace=True)
lilac_114_1_1.columns = ['Time', 'Current', 'Voltage']
lilac_114_1_1.head()

# define tau (user provided), distance ratio threshold R (user defined hyperparameter), and range of D to search over (trainable parameter)
tau = 10
R_ratio = 15 # TODO: tune and try new values of R threshold
D_arr = np.array([1, 2, 4, 6, 8, 10, 12, 15, 18, 20])

# TODO: change window to random sampling data from entire dataset, or start with every 1000th data points and move down on the skip size
# plot code are in the branch "random center robustness" - "MultiFile_FPS_Plot_and_Train_Test_RBF.py" is different in that branch
# TODO: save result as txt file; rows: D values, column: FNN ratio

# create the time delay vectors for each data points with each of the D values
original_data = lilac_114_1_1.to_numpy()
T = original_data[:, 0]
I = original_data[:, 1]
V_0 = original_data[:, 2] # voltage at 0*tau

# time the operation
start = time.time()

# store all the array in a list
time_delay_datasets = [V_0[:-D*tau][:, None] for D in D_arr] # shape is different for each voltage value in the list

for D_index in range(len(D_arr)):
    # print(time_delay_datasets[D_index].shape)
    # print(np.array([V_0[d*tau:-(D_arr[D_index]-d)*tau]
    #     if -(D_arr[D_index]-d)*tau != 0 else V_0[d*tau:]
    #     for d in range(1, 1+D_arr[D_index])]).T.shape)
    # for each D we want to append to the array of data, a new dimension of voltage
    time_delay_datasets[D_index] = np.concatenate(
                                            [time_delay_datasets[D_index],
                                                np.array([V_0[d*tau:-(D_arr[D_index]-d)*tau]
                                                if -(D_arr[D_index]-d)*tau != 0 else V_0[d*tau:]
                                                for d in range(1, 1+D_arr[D_index])]).T],
                                            axis=1)

end = time.time()

print(f"This took {end-start}.")


# change the data from array to tensor for faster calculation
torch.device('cuda')
time_delay_datasets = [torch.tensor(arr) for arr in time_delay_datasets]

for dataset in time_delay_datasets:
    generate_min_dist_datapoints(dataset) # data would be saved as numpy files
"""
Calculate FNN statistics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import time
import torch

def count_fnn(dataset, threshold_R=R):
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
            fnn += 1
        else:
            tnn += 1
    return fnn / (fnn+tnn)

# iterate through the D array and run the ratio calculation algorithm
# from collections import defaultdict
# result = defaultdict(float)
# for d_index in range(len(D_arr)):
#     dataset = time_delay_datasets[d_index] # get the time delay vectors
#     min_dist_vector_pairs = generate_min_dist_datapoints(dataset) # generate data pairs
#     result[D_arr[d_index]] = count_fnn(min_dist_vector_pairs)

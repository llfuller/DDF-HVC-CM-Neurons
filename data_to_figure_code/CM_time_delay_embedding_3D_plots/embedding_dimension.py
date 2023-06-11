import numpy as np
from sklearn.neighbors import NearestNeighbors

# def false_nearest_neighbors(time_series, max_dim=10, tau=1, r=10, metric='euclidean'):
#     def embed_time_series(time_series, dim, tau):
#         embedded = np.array([time_series[i:(i + dim * tau):tau] for i in range(time_series.size - (dim - 1) * tau)])
#         return embedded
#
#     embedded_data = [embed_time_series(time_series, dim, tau) for dim in range(1, max_dim + 1)]
#
#     fnn_ratios = []
#     for dim in range(max_dim - 1):
#         nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric=metric).fit(embedded_data[dim])
#         distances, indices = nbrs.kneighbors(embedded_data[dim])
#
#         neighbor_idxs = indices[:, 1]
#
#         true_neighbor_dists = np.linalg.norm(embedded_data[dim] - embedded_data[dim][neighbor_idxs], axis=-1)
#         next_dim_neighbor_dists = np.linalg.norm(
#             embedded_data[dim + 1][:len(embedded_data[dim])] - embedded_data[dim + 1][neighbor_idxs], axis=-1)
#
#         fnn_ratio = np.sum((next_dim_neighbor_dists / true_neighbor_dists) > r) / len
#         fnn_ratios.append(fnn_ratio)
#
#     embedding_dimension = np.argmin(np.array(fnn_ratios) < 0.1) + 1
#     print(f"FNN Ratio: {fnn_ratio}")
#     dim_dict = {}
#     dim_dict[dim] = fnn_ratio
#     if false_neighbor_ratio <= 0.01:
#         return dim, dim_dict
#
#     return embedding_dimension, {dim + 1: ratio for dim, ratio in enumerate(fnn_ratios)}


def false_nearest_neighbors(time_series, max_dim=10, tau=1, r=10, metric='euclidean', chunk_size=85000):
    def embed(data, dim, tau):
        return np.array([data[i:i + dim * tau:tau] for i in range(len(data) - (dim - 1) * tau)])

    def calculate_distances_chunked(embedded_data_short, chunk_size):
        dists = np.zeros((len(embedded_data_short), len(embedded_data_short)))
        np.fill_diagonal(dists, np.inf)
        for i in range(0, len(embedded_data_short), chunk_size):
            end_i = i + chunk_size
            for j in range(0, len(embedded_data_short), chunk_size):
                end_j = j + chunk_size
                chunk_dists = np.linalg.norm(embedded_data_short[i:end_i, np.newaxis] - embedded_data_short[j:end_j],
                                             axis=-1)
                dists[i:end_i, j:end_j] = chunk_dists
        return dists

    N = len(time_series)
    dim_dict = {}
    for dim in range(1, max_dim + 1):
        # print(f"beginning dim = {dim}")
        embedded_data = embed(time_series, dim + 1, tau)
        embedded_data_short = embedded_data[:-1, :-1]
        # if dim<=3:
        # dists = np.linalg.norm(embedded_data_short[:, None] - embedded_data_short[None, :], axis=-1)
        # else:
        dists = calculate_distances_chunked(embedded_data_short, chunk_size)
        np.fill_diagonal(dists, np.inf)  # Set the diagonal elements to a large value
        neighbor_idxs = np.argmin(dists, axis=1)
        neighbor_dists = dists[np.arange(len(neighbor_idxs)), neighbor_idxs]

        if metric == 'euclidean':
            true_neighbor_dists = np.linalg.norm(embedded_data[:-1] - embedded_data[neighbor_idxs], axis=-1)
        elif metric == 'manhattan':
            true_neighbor_dists = np.sum(np.abs(embedded_data[:-1] - embedded_data[neighbor_idxs]), axis=-1)
        elif metric == 'chebyshev':
            true_neighbor_dists = np.max(np.abs(embedded_data[:-1] - embedded_data[neighbor_idxs]), axis=-1)
        else:
            raise ValueError("Invalid metric")

        false_neighbors = np.sum(true_neighbor_dists / neighbor_dists > r)
        false_neighbor_ratio = false_neighbors / (N - (dim - 1) * tau)
        # print(f"FNN Ratio: {false_neighbor_ratio}")
        dim_dict = {}
        dim_dict[dim]=false_neighbor_ratio
        if false_neighbor_ratio <= 0.01:
            return dim, dim_dict
    # print("Ratio never dropped below 0.01. Report max dimension attempted.")
    return max_dim, dim_dict

#
# # Generate or load your time series data
# # Here, we generate a sample time series using numpy
# # time_series_data = np.loadtxt('phasic/2014_09_10_0013_VIt.txt')[::10,1]
# # print(time_series_data.shape)
# time_series_data = np.loadtxt('tonic/2014_12_11_0017_VIt.txt')[::20,1]
# print(time_series_data.shape)
#
# chunk_size = 20000
# quantity = "I" # current
# units = "pA"
#
# # Calculate the embedding dimension using the FNN method
# print("Beginning calculation")
# embedding_dimension, dim_dict = false_nearest_neighbors(time_series_data, max_dim=5, tau=5, r=5, metric='euclidean', chunk_size=chunk_size)
#
# # Print the result
# print(f"Dimensions and FNN ratio: {dim_dict}")
# print(f"Estimated embedding dimension: {embedding_dimension}")


#================================ODEs Test======================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# def lorenz(t, state, sigma, rho, beta):
#     x, y, z = state
#     return [
#         sigma * (y - x),
#         x * (rho - z) - y,
#         x * y - beta * z
#     ]
# # Lorenz system parameters
# sigma, rho, beta = 10, 28, 8/3
#
# # Initial state and integration time
# initial_state = [-3.1,-3.1, 20.7]
# t_span = (0, 300)
# t_eval = np.linspace(t_span[0], t_span[1], num=10000)
# # Solve the Lorenz system
# solution = solve_ivp(rossler_system, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval)


def rossler_system(t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]
# # Set initial conditions and parameters
# initial_state = [0.1, 0.1, 0.1]
# t_span = (0, 1000)
# t_eval = np.linspace(*t_span, 10000)
# # Solve the Rössler system
# solution = solve_ivp(rossler_system, t_span, initial_state, t_eval=t_eval)
# # Extract the x variable time series
# time_series_data = solution.y[0]
#
# plt.figure()
# plt.plot(time_series_data)
# plt.show()

# def autocorrelation(time_series, max_lag=None):
#     if max_lag is None:
#         max_lag = len(time_series) // 2
#     acf = np.correlate(time_series, time_series, mode='full')[len(time_series) - 1:]
#     return acf[:max_lag] / acf[0]
#
# time_series_data = ...  # Load your time series data
# max_lag = 100
# acf = autocorrelation(time_series_data, max_lag)
#
# # Find the first local minimum of the autocorrelation function
# tau = np.where(np.diff(acf) > 0)[0][0] + 1
# print(f"First local min of tau is {tau}")
# plt.figure()
# plt.plot(acf[:max_lag] < 0+1)
# plt.show()

def mutual_information(time_series, max_lag=None, n_bins=10):
    if max_lag is None:
        max_lag = len(time_series) // 2

    def entropy(counts):
        probs = counts / np.sum(counts)
        return -np.sum(probs * np.log2(probs + np.finfo(float).eps))

    def joint_entropy(counts):
        joint_probs = counts / np.sum(counts)
        return -np.sum(joint_probs * np.log2(joint_probs + np.finfo(float).eps))

    def mutual_information_given_lag(lag):
        time_series_1 = time_series[:-lag]
        time_series_2 = time_series[lag:]

        hist_1, bin_edges_1 = np.histogram(time_series_1, bins=n_bins)
        hist_2, bin_edges_2 = np.histogram(time_series_2, bins=n_bins)

        joint_hist, _, _ = np.histogram2d(time_series_1, time_series_2, bins=(bin_edges_1, bin_edges_2))

        H_1 = entropy(hist_1)
        H_2 = entropy(hist_2)
        H_12 = joint_entropy(joint_hist)

        return H_1 + H_2 - H_12

    return np.array([mutual_information_given_lag(lag) for lag in range(1, max_lag + 1)])

# max_lag = 100
# ami = mutual_information(time_series_data, max_lag)
#
# # Find the first minimum of the average mutual information
# # tau = np.argmin(np.diff(ami) > 0) + 1
# tau = np.argmin(ami)
#
#
# print(f"First local min of tau is {tau}")
# plt.figure()
# plt.plot(ami)
# plt.title("AMI")
# plt.ylabel("AMI")
# plt.xlabel("tau (timesteps)")
# plt.show()
#
# # Calculate the embedding dimension using the FNN method
# dim_list = []
# r_list = np.linspace(1,20,num=20)
# for r in r_list:
#     print(f"Using r={r}")
#     embedding_dimension, dim_dict = false_nearest_neighbors(time_series_data, max_dim=5, tau=tau, r=r, metric='euclidean')
#     # Print the result
#     print(f"Dimensions and FNN ratio: {dim_dict}")
#     print("Estimated embedding dimension:", embedding_dimension)
#     dim_list.append(embedding_dimension)
#
# plt.figure()
# plt.plot(r_list, dim_list)
# plt.title(f"Embedding dimension for Rossler_x (tau={tau})")
# plt.xlabel("r")
# plt.ylabel("FNN Ratio")
# plt.figure()


#=======================Strange Current===========================

# data = np.loadtxt('tonic/2014_12_11_0017_VIt.txt')[::60,1]
# max_lag = 100
# ami = mutual_information(data, max_lag)
# # Find the first minimum of the average mutual information
# tau = np.argmin(ami)
# print(f"First local min of tau is {tau}")
# plt.figure()
# plt.plot(ami)
# plt.title("AMI of Strange Meliza Current")
# plt.ylabel("AMI")
# plt.xlabel("tau (timesteps)")
# plt.show()
#
# # Calculate the embedding dimension using the FNN method
# dim_list = []
# r_list = np.linspace(1,20,num=20)
# for r in r_list:
#     print(f"Using r={r}")
#     embedding_dimension, dim_dict = false_nearest_neighbors(data, max_dim=5, tau=tau, r=r, metric='euclidean')
#     # Print the result
#     print(f"Dimensions and FNN ratio: {dim_dict}")
#     print("Estimated embedding dimension:", embedding_dimension)
#     dim_list.append(embedding_dimension)
# plt.figure()
# plt.plot(r_list, dim_list)
# plt.title(f"Embedding dimension for Strange Meliza Current (Modified L63x?) (tau={tau})")
# plt.xlabel("r")
# plt.ylabel("FNN Ratio")
# plt.figure()

#=================Tonic Neuron Voltage====================================

data = np.loadtxt('tonic/2014_12_11_0017_VIt.txt')[10000:40000,0]
max_lag = 20
ami = mutual_information(data, max_lag)
# Find the first minimum of the average mutual information
tau = 4
print(f"First local min of tau is {tau}")
plt.figure()
plt.plot(ami)
plt.title("AMI of Tonic Meliza Neuron Voltage")
plt.ylabel("AMI")
plt.xlabel("tau (timesteps)")
plt.show()

plt.figure()
plt.plot(data)
plt.title("Meliza Neuron Voltage")
plt.xlabel("time (ms)")
plt.ylabel("Voltage (mV)")
plt.show()

# Calculate the embedding dimension using the FNN method
dim_list = []
r_list = np.linspace(1,20,num=20)
for r in r_list:
    print(f"Using r={r}")
    embedding_dimension, dim_dict = false_nearest_neighbors(data, max_dim=5, tau=tau, r=r, metric='euclidean')
    # Print the result
    print(f"Dimensions and FNN ratio: {dim_dict}")
    print("Estimated embedding dimension:", embedding_dimension)
    dim_list.append(embedding_dimension)
plt.figure()
plt.plot(r_list, dim_list)
plt.title(f"Embedding dimension for Tonic Response to Strange Current (Modified L63x?) (tau={tau})")
plt.xlabel("r")
plt.ylabel("FNN Ratio")
plt.figure()
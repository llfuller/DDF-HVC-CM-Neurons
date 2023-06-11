import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt

def create_time_series_embedding(series, dimension, tau=1):
    n = len(series)
    return np.array([series[i:i + dimension * tau:tau] for i in range(n - (dimension - 1) * tau)])


def false_nearest_neighbors(series, max_dimension, tau=1, threshold=2.0):
    fnn_ratios = []

    for dimension in range(1, max_dimension + 1):
        embedded_data = create_time_series_embedding(series, dimension, tau)
        neighbors = NearestNeighbors(n_neighbors=2).fit(embedded_data)

        distances, indices = neighbors.kneighbors(embedded_data)

        nn_distances = distances[:, 1]
        nn_indices = indices[:, 1]

        next_embedded_data = create_time_series_embedding(series, dimension + 1, tau)
        next_neighbors = NearestNeighbors(n_neighbors=2).fit(next_embedded_data)
        # # Ensure that the indices in nn_indices are within the bounds of the next_embedded_data array
        # valid_nn_indices = nn_indices[nn_indices < len(next_embedded_data)]
        distances_next, indices_next = next_neighbors.kneighbors(next_embedded_data)

        nn_next_distances = distances_next[:, 1]
        nn_next_indices = indices_next[:, 1]

        # distances_next = np.linalg.norm(next_embedded_data - next_embedded_data[valid_nn_indices], axis=1)

        fnn_ratio = np.sum((nn_next_distances / nn_distances[:len(nn_next_indices)]) > threshold) / len(embedded_data)
        fnn_ratios.append(fnn_ratio)

        if fnn_ratio < 0.05 and dimension > 1:
            break

    return fnn_ratios, dimension


# # Sample time series data (replace this with your own data)
# series = np.sin(2 * np.pi * np.arange(100) / 10)
#
# # Set maximum embedding dimension to search
# max_dimension = 10
#
# # Find the minimum embedding dimension using FNN
# fnn_ratios, min_dimension = false_nearest_neighbors(series, max_dimension)
#
# print("FNN Ratios:", fnn_ratios)
# print("Minimum Embedding Dimension:", min_dimension)

def shift_series(series, lag):
    return series[:-lag], series[lag:]

def average_mutual_information(series, max_lag):
    ami_scores = []
    for lag in range(1, max_lag + 1):
        shifted_series1, shifted_series2 = shift_series(series, lag)
        ami = mutual_info_score(shifted_series1, shifted_series2)
        ami_scores.append(ami)
    return ami_scores

def estimate_tau(series, max_lag, threshold_ratio=0.1):
    """Main function to call for tau estimation"""
    ami_scores = average_mutual_information(series, max_lag)
    first_min_ami = np.argmin(ami_scores)
    threshold = ami_scores[0] * threshold_ratio

    for i in range(first_min_ami + 1, max_lag):
        if ami_scores[i] < threshold:
            return i, ami_scores
    return first_min_ami, max_lag, ami_scores

# # Sample time series data (replace this with your own data)
# series = np.sin(2 * np.pi * np.arange(100) / 10)
#
# # Set maximum lag to search
# max_lag = 20
#
# # Estimate tau using AMI method
# tau, ami_scores = estimate_tau(series, max_lag)
#
# print("Estimated Tau:", tau)
#
# # Plot ami_scores against tau
# plt.plot(range(1, max_lag + 1), ami_scores)
# plt.xlabel('Tau')
# plt.ylabel('Average Mutual Information')
# plt.title('Average Mutual Information vs Tau')
# plt.axvline(x=tau, color='r', linestyle='--', label=f'Estimated Tau = {tau}')
# plt.legend()
# plt.show()








# def mutual_information(time_series, max_lag=None, n_bins=10):
#     if max_lag is None:
#         max_lag = len(time_series) // 2
#
#     def entropy(counts):
#         probs = counts / np.sum(counts)
#         return -np.sum(probs * np.log2(probs + np.finfo(float).eps))
#
#     def joint_entropy(counts):
#         joint_probs = counts / np.sum(counts)
#         return -np.sum(joint_probs * np.log2(joint_probs + np.finfo(float).eps))
#
#     def mutual_information_given_lag(lag):
#         time_series_1 = time_series[:-lag]
#         time_series_2 = time_series[lag:]
#
#         hist_1, bin_edges_1 = np.histogram(time_series_1, bins=n_bins)
#         hist_2, bin_edges_2 = np.histogram(time_series_2, bins=n_bins)
#
#         joint_hist, _, _ = np.histogram2d(time_series_1, time_series_2, bins=(bin_edges_1, bin_edges_2))
#
#         H_1 = entropy(hist_1)
#         H_2 = entropy(hist_2)
#         H_12 = joint_entropy(joint_hist)
#
#         return H_1 + H_2 - H_12
#
#     return np.array([mutual_information_given_lag(lag) for lag in range(1, max_lag + 1)])


# def false_nearest_neighbors(time_series, max_dim=10, tau=1, r=10, metric='euclidean', chunk_size=85000):
#     def embed(data, dim, tau):
#         return np.array([data[i:i + dim * tau:tau] for i in range(len(data) - (dim - 1) * tau)])
#
#     def calculate_distances_chunked(embedded_data_short, chunk_size):
#         dists = np.zeros((len(embedded_data_short), len(embedded_data_short)))
#         np.fill_diagonal(dists, np.inf)
#         for i in range(0, len(embedded_data_short), chunk_size):
#             end_i = i + chunk_size
#             for j in range(0, len(embedded_data_short), chunk_size):
#                 end_j = j + chunk_size
#                 chunk_dists = np.linalg.norm(embedded_data_short[i:end_i, np.newaxis] - embedded_data_short[j:end_j],
#                                              axis=-1)
#                 dists[i:end_i, j:end_j] = chunk_dists
#         return dists
#
#     N = len(time_series)
#     dim_dict = {}
#     for dim in range(1, max_dim + 1):
#         # print(f"beginning dim = {dim}")
#         embedded_data = embed(time_series, dim + 1, tau)
#         embedded_data_short = embedded_data[:-1, :-1]
#         # if dim<=3:
#         # dists = np.linalg.norm(embedded_data_short[:, None] - embedded_data_short[None, :], axis=-1)
#         # else:
#         dists = calculate_distances_chunked(embedded_data_short, chunk_size)
#         np.fill_diagonal(dists, np.inf)  # Set the diagonal elements to a large value
#         neighbor_idxs = np.argmin(dists, axis=1)
#         neighbor_dists = dists[np.arange(len(neighbor_idxs)), neighbor_idxs]
#
#         if metric == 'euclidean':
#             true_neighbor_dists = np.linalg.norm(embedded_data[:-1] - embedded_data[neighbor_idxs], axis=-1)
#         elif metric == 'manhattan':
#             true_neighbor_dists = np.sum(np.abs(embedded_data[:-1] - embedded_data[neighbor_idxs]), axis=-1)
#         elif metric == 'chebyshev':
#             true_neighbor_dists = np.max(np.abs(embedded_data[:-1] - embedded_data[neighbor_idxs]), axis=-1)
#         else:
#             raise ValueError("Invalid metric")
#
#         false_neighbors = np.sum(true_neighbor_dists / neighbor_dists > r)
#         false_neighbor_ratio = false_neighbors / (N - (dim - 1) * tau)
#         # print(f"FNN Ratio: {false_neighbor_ratio}")
#         dim_dict = {}
#         dim_dict[dim]=false_neighbor_ratio
#         if false_neighbor_ratio <= 0.01:
#             return dim, dim_dict
#     # print("Ratio never dropped below 0.01. Report max dimension attempted.")
#     return max_dim, dim_dict

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def estimate_Ridge_Regression_parameter(datasets_collection):
    # # Define the parameter grid
    # param_grid = {
    #     'C': [0.1, 1, 10, 100],
    #     'kernel': ['linear', 'rbf'],
    #     'gamma': ['scale', 'auto']
    # }
    #
    # # Create an instance of the estimator (SVC in this case)
    # svc = SVC()
    #
    # # Create the GridSearchCV object
    # grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
    #
    # # Fit the GridSearchCV object to the training data
    # grid_search.fit(X_train, y_train)
    #
    # # Get the best hyperparameters
    # best_params = grid_search.best_params_
    # print("Best hyperparameters:", best_params)
    #
    # # Get the best estimator
    # best_estimator = grid_search.best_estimator_
    #
    # # Evaluate the best estimator on the test data
    # test_accuracy = best_estimator.score(X_test, y_test)
    # print("Test accuracy:", test_accuracy)
    # return

    V_time_delayed = datasets_collection["training"]["concatenated"]["time_delayed"]["V_time_delayed"]

    # Create X with dimensions (N, N_c+1)
    X = gaussian_centers_and_I_stim(x_embedded, centers_matrix, R, I_n, I_np1_padded)  # dim(X)=(N, N_c+1)

    # Split X and Y into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=num_test_timesteps, random_state=2023, shuffle=False)

    # Perform Ridge Regression with cross-validation
    ridge_cv = RidgeCV(alphas=candidate_beta_list, cv=cv_folds)  # Replace alphas with your desired range of regularization parameters
    ridge_cv.fit(X_train, Y_train)

    # Get the best regularization parameter and coefficients
    beta = ridge_cv.alpha_
    W = ridge_cv.coef_
    print("Training finished")
    return [x_embedded, X_train, X_test, Y_train, Y_test, W, beta, centers_matrix, num_test_timesteps]
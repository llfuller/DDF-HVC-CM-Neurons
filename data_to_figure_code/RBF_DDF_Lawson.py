from numba import njit
import make_time_delay_embedding
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import make_plots_3d_coefficient_scatter

# @njit
def gaussian_centers_and_I_stim(x_embedded, centers_matrix, R, I_n, I_np1_padded):
    """
    Calculate {e^-R(x-centers)^2, I(n) + I(n-1)

    Args:
    x_embedded (numpy array): A (D_E, N) dimensional matrix, where D_E is the embedding dimension and N is the number of time steps.
    centers_matrix (numpy array): A (D_E, N_c) dimensional matrix, where N_c is the number of columns.
    W (numpy array): An (N_c+1)-dimensional vector.
    R (float or numpy array): A scalar or N-dimensional vector for scaling the distance in the f() function.
    I_n (numpy array): A (D_E, N)-dimensional array representing I(t).
    I_np1_padded (float): A (D_E, N)-dimensional array representing I(t+1).

    Returns:
    numpy array: X. dim(X)=(N, N_c+1)
    """
    # Compute distances for all time steps and columns at once
    D_E = x_embedded.shape[0]
    N   = x_embedded.shape[1]
    N_c = centers_matrix.shape[1]
    distances = np.linalg.norm(x_embedded[:, :, np.newaxis] - centers_matrix[:, np.newaxis, :], axis=0)  # dim(distances)=(N, N_c)
    # Compute e_values for all time steps and columns at once
    e_values = np.exp(-R * distances**2)  # dim(e_values)=(N, N_c)

    # Create f_values matrix with the shape (N, N_c+1)
    X = np.zeros((e_values.shape[0], N_c + 1))
    # Assign e_values to the first N_c columns
    X[:, :N_c] = e_values
    # Assign I_n + I_np1 term to the N_c+1 column
    X[:, N_c] = I_n + I_np1_padded

    return X  # dim(centers_and_I_stim)=(N, N_c+1)

@njit
def calculate_delta_x(W, X):
    """
    Calculate the sum of f(|x-S_#|) for each column in S_matrix for each time step in x_embedded.

    Args:
    X (numpy array): An (N, N_c+1)-dimensional array
    W (numpy array): An (N_c+1)-dimensional vector.

    Returns:
    numpy array: The sum of f(|x-S_#|) for each column in S_matrix for each time step in x_embedded; dim(result)=(N,)
    """
    delta_x = W @ X.T  # dim(result)=(N,)
    return delta_x  # dim(X)=(N,)

def train(V, I, T, D_E, R, train_start_timestep, train_end_timestep, num_train_timesteps, num_test_timesteps, N_c=500, candidate_beta_list = np.logspace(-6,6,13), cv_folds=5):
    print("Training")
    # Create time delay embedding for x
    x_embedded = make_time_delay_embedding.time_delay_embedding(V, T, D_E) # dim(x_embedded) = (D_E, N)
    x_embedded_training = x_embedded[:,train_start_timestep:train_end_timestep]

    # Create time delay embedding for I
    I_n, I_np1 =  I[train_start_timestep:train_end_timestep], I[train_start_timestep+1:train_end_timestep] # I_n dimension (D_E, N); I_np1 dimension (D_E, N-1)
    # Extend I_np1 to have the same dimensions as I_n
    I_np1_padded = np.pad(I_np1, (0, 1), mode='edge') # pad with the last column (N-2)

    # Create Y from x_embedded
    Y = x_embedded_training[0, 1:]  # Y represents x(n+1) values
    Y = np.pad(Y, (0, 1), mode='edge') # pad with the last column (N-2)

    N = x_embedded_training.shape[1]

    print("Performing k-means")
    # Example usage with x_embedded:
    # Apply k-means clustering to x_embedded in training section
    kmeans = KMeans(n_clusters=N_c, random_state=42)
    kmeans.fit(x_embedded_training.T)  # Note that we transpose x_embedded to have the data points in rows

    # Get the cluster centroids (centers)
    centers_matrix = kmeans.cluster_centers_.T  # Transpose back to have dim(centers)=(D_E,N_c)
    print("Centers created")


    # Create X with dimensions (N, N_c+1)
    X = gaussian_centers_and_I_stim(x_embedded_training, centers_matrix, R, I_n, I_np1_padded)  # dim(X)=(N, N_c+1)

    # Split X and Y into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=num_test_timesteps, random_state=2023, shuffle=False)

    # Perform Ridge Regression with cross-validation
    ridge_cv = RidgeCV(alphas=candidate_beta_list, cv=cv_folds)  # Replace alphas with your desired range of regularization parameters
    ridge_cv.fit(X_train, Y_train)

    # Get the best regularization parameter and coefficients
    beta = ridge_cv.alpha_
    W = ridge_cv.coef_
    print("Training finished")
    return [x_embedded_training, X_train, X_test, Y_train, Y_test, W, beta, centers_matrix, num_test_timesteps]


def test(W, x_embedded_start, D_E, I_n, I_np1_padded, test_start_timestep, test_end_timestep, centers_matrix, R):
    """

    Returns:

    """
    print("Testing")
    num_test_timesteps = test_end_timestep - test_start_timestep
    # Preallocate Y_pred as a numpy array with the size of the last third of the length of X_train
    x_pred = np.zeros((D_E, num_test_timesteps)) # dim(x_pred) = (D_E, num_test_timesteps)
    x_pred[:,0] = x_embedded_start # dim(x_pred) = (D_E, num_test_timesteps)
    test_timestep_array = np.arange(start=test_start_timestep, stop=test_end_timestep) # dim(timestep_array) = (num_test_timesteps)

    # Iterate over the indices of X_train
    for n in range(num_test_timesteps-1):
        # print(f"n={n}")
        X_n = gaussian_centers_and_I_stim((x_pred[:,n])[:,np.newaxis], centers_matrix, R, I_n[n:n+1], I_np1_padded[n:n+1])
        x_pred[1:, n + 1] = x_pred[0:-1, n + 1] # update past components
        # Calculate delta_x for each time step and update the corresponding element in Y_pred
        x_pred[0,n+1] = x_pred[0,n] + calculate_delta_x(W, X_n)#.reshape(1,*(X_n[:,n]))) # scalar
    print("Testing finished")
    return x_pred, test_timestep_array

data_filepath = "HVC_x_Red171_Neuron2_Epoch1_good_prediction/epoch_1.txt"
data = np.loadtxt(data_filepath) # dim(data) = (N_tot, 2)
I_loaded = data[:,0] # dim(I_loaded) = N_tot
V_loaded = data[:,1] # dim(V_loaded) = N_tot
dt = 0.00002 # seconds (0.02 ms)
T = 10 # time delay in number of timesteps (not seconds)
D_E = 3 # time delay embedding dimensions
R = 10**(-3) # 1/(2*sigma^2) of the gaussian center
N_total = I_loaded.shape[0] # number of timesteps in original data
train_start_timestep = 0
train_end_timestep = int(2 * float(N_total) / 3)
num_train_timesteps = train_end_timestep-train_start_timestep
test_start_timestep = int(2 * float(N_total) / 3)
test_end_timestep = N_total
num_test_timesteps = test_end_timestep - test_start_timestep
timestep_array = np.array(range(train_start_timestep, test_end_timestep))

N_c = 100 # number of centers
candidate_beta_list = np.logspace(-9,-3,6) # regularization parameters to try
cv_folds=5 # number of cross-validation k-fold

# Train
x_embedded_training, X_train, X_test, Y_train, Y_test, W, beta, centers_matrix, num_test_timesteps_embedded = train(V_loaded, I_loaded, T, D_E, R, train_start_timestep, train_end_timestep, num_train_timesteps, num_test_timesteps, N_c=N_c, candidate_beta_list = candidate_beta_list, cv_folds=5)

# Create time delay embedding for I
I_n, I_np1 = I_loaded[0:], I_loaded[1:]  # I_n dimension (D_E, N); I_np1 dimension (D_E, N-1)
# Extend I_np1 to have the same dimensions as I_n
I_np1_padded = np.pad(I_np1, (0, 1), mode='edge')  # pad with the last column (N-2)

# Test
x_pred, test_timestep_array = test(W, x_embedded_training[:,-1], D_E, I_n, I_np1_padded, test_start_timestep, test_end_timestep, centers_matrix, R)

# Plot predicted voltage against true voltage
plt.figure()
plt.plot(test_timestep_array*dt,x_pred[0], color="r", linewidth=0.4, label = "V_predicted")
plt.plot(timestep_array[test_start_timestep:test_end_timestep]*dt, V_loaded[test_start_timestep:test_end_timestep], color="black", linewidth=0.4, label="V_truth")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.show()

# make_plots_3d_coefficient_scatter.plot_time_delay_center_values(data_filepath)

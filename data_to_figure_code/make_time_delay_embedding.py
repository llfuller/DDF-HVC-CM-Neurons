import numpy as np
from numba import njit

@njit
def time_delay_embedding(data, T, D_E):
    """
    Create a time delay embedding of the data.

    Args:
    data (numpy array): 1D array of data points.
    T (int): Time delay.
    D_E (int): Embedding dimension.

    Returns:
    numpy array: Time delay embedded data of shape (D_E, N), where N is the number of columns.
    """
    N = len(data) - (D_E - 1) * T
    embedded_data = np.zeros((D_E, N))

    for d in range(D_E):
        embedded_data[D_E - 1 - d] = data[d * T : d * T + N]

    return embedded_data

def time_delay_embedded_time_series(t_array, T, D_E):
    """
    Create a time delay embedding of the time array.

    Args:
        t_array (numpy array): 1D array of time points.
        T (int): Time delay.
        D_E (int): Embedding dimension.

    Returns:
        numpy array: Time delay embedded time array of shape (N,), where N is the number of elements in the resulting embedded time array.
    """
    return_array = t_array[(D_E-1)*T:]

    return return_array

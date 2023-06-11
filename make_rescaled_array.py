import numpy as np

def rescale_array(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Rescale the elements of a numpy array so that the min value is `low` and the max value is `high`.

    Args:
        arr (np.ndarray): Input numpy array.
        low (float): The new minimum value for the array.
        high (float): The new maximum value for the array.

    Returns:
        np.ndarray: Rescaled numpy array.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)

    if min_val == max_val:
        raise ValueError("The input array contains the same values. Rescaling is not possible.")

    rescaled_arr = (arr - min_val) * (high - low) / (max_val - min_val) + low
    return rescaled_arr

def main():

    # All the following is example usage:
    # Assuming the rescale_array function is already defined or imported

    # Define N
    N = 5

    # Create arrays containing mins and maxes of length N
    mins = np.array([0, 1, 2, 3, 4])
    maxes = np.array([5, 6, 7, 8, 9])

    # Define I_stim as an example array
    I_stim = np.array([2, 4, 6, 8, 10])

    # Loop over range 0 to N-1
    for n in range(N):
        # Select the nth element from mins and maxes arrays to form a pair
        min_val = mins[n]
        max_val = maxes[n]

        # Call rescale_array function with I_stim, min_val, and max_val
        scaled_I_stim = rescale_array(I_stim, min_val, max_val)

        # Print or use the scaled_I_stim as needed
        print(f"Scaled I_stim for min={min_val} and max={max_val}: {scaled_I_stim}")

if __name__ == '__main__':
    main()

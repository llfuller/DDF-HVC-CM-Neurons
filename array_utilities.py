import numpy as np
import copy

def replace_ranges_with_nan(t_arr, range_list=None):
    """
    Returns t_arr but with any indices in range_list set to nan
    """
    return_arr = copy.copy(t_arr)
    if range_list != None:
        for a_range in range_list:
            return_arr[a_range] = np.nan
    return return_arr

def remove_elements(t_arr, range_list):
    """
    Returns t_arr but with any indices in range_list removed
    """
    time_indices_to_be_removed = np.zeros((t_arr.shape[0])).astype(bool)
    for t_index, t in enumerate(time_indices_to_be_removed):
        for a_range in range_list:
            if t_index>a_range[0] and t_index<a_range[-1]:
                time_indices_to_be_removed[t_index] = True
    t_arr_shortened = np.delete(t_arr,obj=time_indices_to_be_removed,axis=0)
    return t_arr_shortened

def set_range(arr, lower_bound, upper_bound):
    """
    Returns a version of the input array scaled to have values between lower_bound and upper_bound.
    """
    return np.divide(arr+np.abs(np.min(arr)),np.max(arr)+np.abs(np.min(arr)))*\
           (upper_bound + np.abs(lower_bound)) - np.abs(lower_bound)

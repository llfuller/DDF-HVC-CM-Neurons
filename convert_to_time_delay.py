import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy


def print_div(a_str):
    # print string with line underneath it and automatically start a new line
    return print(f"{a_str}\n========================\n")

def convert_to_time_delay(arr, D_E, len_skips, num_tsteps_in_time_delay_vector, timestep_start):
    """
    Converts array to time delay embedded version using general tau, D_E, and num_tsteps_in_time_delay_vector.
    :param arr: numpy array to convert to time delay vector
    :param D_E: number of time delay components in vector; dimension of embedding space
    :param len_skips: number of timesteps between dimension 1 and dimension 2 of time delay vector
    :param num_tsteps_in_time_delay_vector: number of timesteps in time delay vector
    :param timestep_start: starting timestep of most current (top row, 0 time delay) of time delay matrix
    :return: td_arr: numpy array; dim()=(D_E,num_tsteps_in_time_delay_vector). Row 0 (0,num_tsteps...) is most recent.
            Rest are past.
            # Example:
            # first row represents no time delay, should be [50, 51, 52, ...]
            # second row represents one time delay, should be [47, 48, 49, ...]
            # there should be 5 rows.
    """
    td_arr = np.zeros((D_E, num_tsteps_in_time_delay_vector))
    # first row represents no time delay, should be [50, 51, 52, ...]
    # second row represents one time delay, should be [47, 48, 49, ...]
    # there should be D_E rows.
    timestep_end = timestep_start + num_tsteps_in_time_delay_vector # for present time
    assert (timestep_end < len(arr))
    print_div(f"timestep_start - (D_E - 1) * len_skips: {timestep_start - (D_E - 1) * len_skips}")
    assert (timestep_start - (D_E - 1) * len_skips >= 0)
    timestep_start = int(timestep_start)
    timestep_end = int(timestep_end)
    for d in range(D_E):
        print(f"d={d}")
        print(f"timestep_start={timestep_start}")
        print(f"timestep_end={timestep_end}")
        print(f"len_skips={len_skips}")
        print(f"timestep_start - d * len_skips={timestep_start - d * len_skips}")
        print(f"timestep_end - d * len_skips={timestep_end - d * len_skips}")

        td_arr[d] = arr[range(timestep_start - d * len_skips, timestep_end - d * len_skips)]
    return td_arr

def calcluate_min_timestep_start_assuming_D_E_and_len_skips(D_E, len_skips):
    """
    :param D_E: number of time delay components in vector; dimension of embedding space
    :param len_skips: number of timesteps between dimension 1 and dimension 2 of time delay vector
    :return: minimum starting time allowable
    # Example:
    # assume arr = [0,1,2,3,4,5,6,7,8,9,10]
    # D_E = 3
    # len_skips = 2
    # num_tsteps_in_time_delay_vector=3
    # td_arr = [[4,5,6]
    #           [2,3,4]
    #           [0,1,2]]
    """
    # starting point here is (D_E-1)*len_skips away
    return (D_E-1)*len_skips

def calcluate_max_timestep_start_assuming_D_E_and_len_skips(arr, num_tsteps_in_time_delay_vector):
    """
    :param arr: numpy array to convert to time delay vector
    :param num_tsteps_in_time_delay_vector: number of timesteps in time delay vector
    :return: max starting time allowable
    # Example:
    # assume arr = [0,1,2,3,4,5,6,7,8,9,10]
    # len(arr) = 11
    # D_E = 3
    # len_skips = 2
    # num_tsteps_in_time_delay_vector=7
    # td_arr = [[4,5,6,7,8,9,10]
    #           [2,3,4,...]
    #           [0,1,2,...]]
    # starting point here is at index  len(arr)-num_tsteps_in_time_delay_vector + 1 (should be index 5)
    """
    return len(arr)-num_tsteps_in_time_delay_vector + 1

def plot_3D_time_delay_vector(td_arr, title):
    """
    3-D plotting function for later
    :param td_arr: numpy array; dim()=(D_E,num_tsteps_in_time_delay_vector). Row 0 (0,num_tsteps...) is most recent.
    :param title: type:string
    :return: figure object fig
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(td_arr[0], td_arr[1], td_arr[2], linewidth=0.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title(title)
    plt.show()
    return fig

def return_L63_states():
    rho=28.0
    sigma=10.0
    beta=8.0 / 3.0
    def dfdt_L63(state, t):
        """To be used in odeint"""
        # runs forward in time from 0, cannot just compute at arbitrary t
        x, y, z = state  # Unpack the state vector
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        dstatedt = np.array([dxdt,dydt,dzdt])
        return dstatedt
    state0 = [-3.1, -3.1, 20.7]
    start_time=0
    end_time=1000
    num_timesteps=1000000
    # numpy array of times at which you want to calculate the state variables
    times_array = np.linspace(start=start_time, stop=end_time, num=num_timesteps)
    states = odeint(dfdt_L63, state0, times_array)
    return states

def return_Colpitts_states():
    # Try with Colpitts:
    alpha = 5.0
    gamma = 0.0797  # 0.08
    q = 0.6898  # 0.7
    eta = 6.273  #6.3
    def dfdt_Colpitts(state, t):
        """To be used in odeint"""
        # runs forward in time from 0, cannot just compute at arbitrary t
        x1, x2, x3 = state  # Unpack the state vector
        dstatedt = alpha * x2, -gamma * (x1 + x3) - q * x2, eta * (x2 + 1 - np.exp(-x1))  # Derivatives
        return dstatedt

    state0 = np.array([0.1, 0.1, 0.1])
    start_time=0
    end_time=100
    num_timesteps=1000000
    times_array = np.linspace(start=start_time, stop=end_time, num=num_timesteps) #numpy array of times at which you want to calculate the state variables
    states = odeint(dfdt_Colpitts, state0, times_array)
    return states

def calculate_axis_means(arr):
    """
    :param arr: dim(arr)=(time,space)
    :return: np.mean(arr) with dim=(space)
    """
    return np.mean(arr,axis=0)

def calculate_axis_moments(arr):
    """
    :param arr: dim(arr)=(time,space)
    :return: np.mean(arr) with dim=(space)
    """
    # first and second moments
    return [scipy.stats.moment(arr,moment=2,axis=0),scipy.stats.moment(arr,moment=3,axis=0)]

def run_examples():
    arr = np.array(range(1,101))
    print_div(arr)

    D_E = 5# embedding dimension
    tau = 3 # number of timesteps to skip
    num_tsteps_in_time_delay_vector = 10

    td_arr = np.zeros((D_E,num_tsteps_in_time_delay_vector))
    # first row represents no time delay, should be [50, 51, 52, ...]
    # second row represents one time delay, should be [47, 48, 49, ...]
    # there should be 5 rows.

    timestep_start = 50 # where does most recent (first row) t-delay vector start?
    timestep_end = timestep_start + num_tsteps_in_time_delay_vector

    # Make sure bound are contained entirely in array
    print_div("length of arr is {len(arr)}")
    assert(timestep_end < len(arr))
    assert(timestep_end - (D_E-1) * tau >= 0)

    for d in range(D_E):
        td_arr[d] = arr[range(timestep_start - d * tau, timestep_end - d * tau)]

    print(td_arr)

    # Now run a function which does this for general tau, D_E, and num_tsteps_in_time_delay_vector:
    print("Running function.")

    arr = np.array(range(0,1000))
    D_E = 5
    len_skips = 10
    td_arr = convert_to_time_delay(arr=arr,
                                  D_E=D_E,
                                  len_skips=len_skips,
                                  num_tsteps_in_time_delay_vector=10,
                                  timestep_start=D_E*len_skips)
    print_div(td_arr)
    print("Ran without issues.")

    print(calcluate_max_timestep_start_assuming_D_E_and_len_skips(arr=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                                                  num_tsteps_in_time_delay_vector=7))

    """L63"""
    # convert x-component only to time delay vector
    states = return_L63_states() # dim = (time,space)
    x_arr = states[:,0] # project to one dimension
    D_E = 3
    len_skips = 70
    td_arr = convert_to_time_delay(arr=x_arr,
                                   D_E=D_E,
                                   len_skips=len_skips,
                                   num_tsteps_in_time_delay_vector=700000,
                                   timestep_start=D_E*len_skips)

    # Now plot
    plot_3D_time_delay_vector(td_arr, "L63 Time Delayed")
    # plot original state space for comparison
    plot_3D_time_delay_vector(states.transpose(), "L63 Original")

    """Colpitts"""
    states = return_Colpitts_states()
    # convert x1-component only to time delay vector
    x_arr = states[:, 0]
    D_E = 3
    len_skips = 20000
    # convert to time delay embedding space:
    td_arr = convert_to_time_delay(arr=x_arr,
                                   D_E=D_E,
                                   len_skips=len_skips,
                                   num_tsteps_in_time_delay_vector=900000,
                                   timestep_start=D_E * len_skips)

    # Now plot time-delay version
    plot_3D_time_delay_vector(td_arr, "Colpitts Time Delayed")

    # plot original state space for comparison
    plot_3D_time_delay_vector(states.transpose(), "Colpitts Original")
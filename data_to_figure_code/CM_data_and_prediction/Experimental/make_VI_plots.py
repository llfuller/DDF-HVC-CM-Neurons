import numpy as np
import matplotlib.pyplot as plt

# Set the properties for using LaTeX and a serif font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def time_series_plot_and_save(data, delta_t, save_folder, title, save_filename, xlabel="Time (s)", ylabel="Amplitude", color='red', vline_x=None):


    """
    Plots and saves a time series plot from a 1-D numpy array `data`.

    Parameters:
    -----------
    data : numpy.ndarray
        1-D numpy array containing the time series data.
    delta_t : float
        Time step between data points in milliseconds.
    save_folder : str
        Folder path to save the plot and data files.
    title : str
        Title of the plot.
    save_filename : str
        Filename to save the plot and data files (without extension).
    xlabel : str, optional
        X-axis label, default is "Time (s)".
    ylabel : str, optional
        Y-axis label, default is "Amplitude".

    Returns:
    --------
    None
    """
    plt.figure()
    time = np.arange(len(data)) * delta_t * 1e-3  # Convert delta_t from ms to s

    plt.plot(time, data, color=color, linewidth=0.5)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if vline_x is not None:
        plt.axvline(x=vline_x, color='green', linewidth=1)

    ax = plt.gca()
    ax.set_facecolor("white")
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', length=6, width=1, color='black', labelsize=12)

    if data.max() >= 1e3 or data.min() <= 1e-3:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    plt.tight_layout()

    # Save the plot and data
    plt.savefig(save_folder + save_filename + ".png")
    plt.savefig(save_folder + save_filename + ".pdf")
    plt.close()

# Load data from files
data1 = np.loadtxt("2014_09_10_0013_VIt.txt") # Phasic CM
data2 = np.loadtxt("2014_12_11_0017_VIt.txt") # Tonic CM

# Separate columns for voltage, current, and time
voltage1, current1, time1 = data1[:, 0], data1[:, 1], data1[:, 2]
voltage2, current2, time2 = data2[:, 0], data2[:, 1], data2[:, 2]

# Calculate delta_t for each dataset (assuming constant time step)
delta_t1 = (time1[1] - time1[0]) * 1000  # Convert from seconds to milliseconds
delta_t2 = (time2[1] - time2[0]) * 1000

# Folder to save the plots
save_folder = "./"

# Plot and save voltage and current for each dataset
time_series_plot_and_save(voltage1, delta_t1, save_folder, "Phasic V(t)", "2014_09_10_0013_V", ylabel="Voltage (mV)", color='blue', vline_x=10.0)
time_series_plot_and_save(current1, delta_t1, save_folder, "Phasic I(t)", "2014_09_10_0013_I", ylabel="Current (pA)", color='red', vline_x=10.0)

time_series_plot_and_save(voltage2, delta_t2, save_folder, "Tonic V(t)", "2014_12_11_0017_V", ylabel="Voltage (mV)", color='blue', vline_x=10.0)
time_series_plot_and_save(current2, delta_t2, save_folder, "Tonic I(t)", "2014_12_11_0017_I", ylabel="Current (pA)", color='red', vline_x=10.0)


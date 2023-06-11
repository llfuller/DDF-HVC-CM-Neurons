import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytinytex
import os  # Add this import

# Set the properties for using LaTeX and a serif font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def time_series_plot_and_save(data, delta_t, save_folder, title, save_filename, t_array=None, xlabel="Time (s)", ylabel="Amplitude", show_plot = False):
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
    if t_array is None:
        time = np.arange(len(data)) * delta_t * 1e-3  # Convert delta_t from ms to s
    if t_array is not None:
        time = t_array

    color = "black"
    if "V" in  ylabel or "Voltage" in ylabel:
        color = "blue"
    if "I" in ylabel or "Current" in ylabel:
        color = "red"
    plt.plot(time, data, color=color, linewidth=0.5)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

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

    # Create the directory if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # Create the directory if it doesn't exist
    if not os.path.exists(save_folder+ "pdf_plot_versions/"):
        os.makedirs(save_folder+ "pdf_plot_versions/")

    # Save the plot and data
    plt.savefig(save_folder + save_filename + ".png")
    plt.savefig(save_folder + "pdf_plot_versions/"+save_filename + ".pdf")
    plt.close()

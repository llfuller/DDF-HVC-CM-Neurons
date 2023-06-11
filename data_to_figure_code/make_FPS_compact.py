import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import make_general_plot

# Set up LaTeX and font settings for plot labels
matplotlib.rc('text', usetex=True)
plt.rc('font', family='serif')


# Function to plot and save the Fourier Power Spectrum
def Fourier_Power_Spectrum_plot_and_save(data, name, sampling_rate, save_folder, title, save_filename, xlim=175):
    """
    Plots and saves the Fourier Power Spectrum of the input data with the specified stylization.

    Parameters
    ----------
    data : numpy array
        Input data to compute the Fourier Power Spectrum.
    name : str
        Name of the data, used for internal purposes.
    sampling_rate : int
        Sampling rate of the input data.
    save_folder : str
        Directory path to save the generated plot and text files.
    title : str
        Title for the plot.
    save_filename : str
        Filename for saving the plot and text files (without file extension).
    xlim : int, optional, default: 175
        Maximum value for the x-axis limit.

    Returns
    -------
    None
    """
    # Perform Fourier Transform, compute power spectrum, and prepare frequency axis
    fourier_transform = np.fft.rfft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))

    # Create the plot
    plt.figure()
    freq_without_0_index = frequency[1:]
    normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))

    # Plot the power spectrum
    plt.fill_between(freq_without_0_index, 0, normalized_power_spec_without_0_index, color='black', alpha=0.5)
    plt.plot(freq_without_0_index, normalized_power_spec_without_0_index, color='black', linewidth=0.5)

    # Set plot labels and limits
    plt.title(title, fontsize=16)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Normalized Power (1.0 = max of spectrum indices [1:])", fontsize=14)
    plt.xlim(0, xlim)

    # Customize plot appearance
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', length=6, width=1, color='black', labelsize=12)

    # Adjust y-axis label format for large or small values
    if normalized_power_spec_without_0_index.max() >= 1e3 or normalized_power_spec_without_0_index.min() <= 1e-3:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    # Create the directory if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print(f"Saving to: {save_folder}{save_filename}")
    # Save and close the plot
    plt.tight_layout()
    plt.savefig(save_folder + save_filename + ".png")
    plt.savefig(save_folder + save_filename + ".pdf")
    np.savetxt(save_folder + save_filename + ".txt",
               np.column_stack((freq_without_0_index, normalized_power_spec_without_0_index)))
    plt.close()

def main():

    # Define a list of dictionaries containing file information
    file_info_list = [
        {
            'filepath': 'I_stimuli/I_colpitts_x_time_dilation=0.2range=300_(I).txt',
            'sampling_rate': 50000,
            'save_folder': 'I_stimuli/',
            'plot_title': "I_{Colpitts-x};\ td=0.2",
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "I_{stim}\ (pA)",
            'index': None,
            'time_indices': "all",
            'xlim': 30,
        },
        {
            'filepath': 'I_stimuli/I_colpitts_x_time_dilation=0.5range=300_(I).txt',
            'sampling_rate': 50000,
            'save_folder': 'I_stimuli/',
            'plot_title': "I_{Colpitts-x};\ td=0.5",
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "I_{stim}\ (pA)",
            'index': None,
            'time_indices': "all",
            'xlim': 30,
        },
        {
            'filepath': 'I_stimuli/I_L63_x_time_dilation=0.5range=300_(I).txt',
            'sampling_rate': 50000,
            'save_folder': 'I_stimuli/',
            'plot_title': "I_{Lorenz63-x};\ td=0.5",
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "I_{stim}\ (pA)",
            'index': None,
            'time_indices': "all",
            'xlim': 30,
        },
        {
            'filepath': 'CM_data_and_prediction/Experimental/2014_09_10_0013_VIt.txt',
            'sampling_rate': 40000,
            'save_folder': 'CM_data_and_prediction/Experimental/',
            'plot_title': "I_{stim}(t)",
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "I_{stim}\ (pA)",
            'index': 1,  # 32425a75 Phasic
            'time_indices': "all",
            'xlim': 60,
        },
        {
            'filepath': 'CM_data_and_prediction/Experimental/2014_12_11_0017_VIt.txt',
            'sampling_rate': 40000,
            'save_folder': 'CM_data_and_prediction/Experimental/',
            'plot_title': "I_{stim}(t)",
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "I_{stim}\ (pA)",
            'index': 1,  # 920061fe Tonic
            'time_indices': "all",
            'xlim': 60,
        },
        {
            'filepath': 'CM_data_and_prediction/Experimental/2014_09_10_0013_VIt.txt',  # Update the filepath if needed
            'sampling_rate': 40000,
            'save_folder': 'CM_data_and_prediction/Experimental/',
            'plot_title': "V(t)",  # Update the plot title for the voltage plot
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "V\ (mV)",  # Update the y-axis label to reflect voltage
            'index': 0,  # Assuming the voltage data is in the first column
            'time_indices': "all",
            'xlim': 60,
        },
        {
            'filepath': 'CM_data_and_prediction/Experimental/2014_12_11_0017_VIt.txt',  # Update the filepath if needed
            'sampling_rate': 40000,
            'save_folder': 'CM_data_and_prediction/Experimental/',
            'plot_title': "V(t)",  # Update the plot title for the voltage plot
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "V\ (mV)",  # Update the y-axis label to reflect voltage
            'index': 0,  # Assuming the voltage data is in the first column
            'time_indices': "all",
            'xlim': 60,
        },
        {
            'filepath': 'Eric_figures_(3D_weight_distribution)/Red171_Neuron1_Epoch1.txt',  # Update the filepath if needed
            'sampling_rate': 50000,
            'save_folder': 'Eric_figures_(3D_weight_distribution)/',
            'plot_title': "I_{stim}(t)",  # Update the plot title for the voltage plot
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "I_{stim}\ (pA)",  # Update the y-axis label to reflect voltage
            'index': 0,  # Assuming the voltage data is in the first column
            'time_indices': "all",
            'xlim': 100,
        },
        {
            'filepath': 'Eric_figures_(3D_weight_distribution)/Red171_Neuron1_Epoch1.txt',  # Update the filepath if needed
            'sampling_rate': 50000,
            'save_folder': 'Eric_figures_(3D_weight_distribution)/',
            'plot_title': "V(t)",  # Update the plot title for the voltage plot
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "V\ (mV)",  # Update the y-axis label to reflect voltage
            'index': 1,  # Assuming the voltage data is in the first column
            'time_indices': "all",
            'xlim': 100,
        },
        {
            'filepath': 'Will_figures_(Spiking_Problems)/ORIGINAL-0001.txt',  # Update the filepath if needed
            'sampling_rate': 50000,
            'save_folder': 'Will_figures_(Spiking_Problems)/',
            'plot_title': "I_{stim}(t)",  # Update the plot title for the voltage plot
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "I_{stim}\ (pA)",  # Update the y-axis label to reflect voltage
            'index': 1,  # Assuming the voltage data is in the first column
            'time_indices': "all", #[10000,125000],
            'xlim': 100,
        },
        {
            'filepath': 'Will_figures_(Spiking_Problems)/ORIGINAL-0001.txt',  # Update the filepath if needed
            'sampling_rate': 50000,
            'save_folder': 'Will_figures_(Spiking_Problems)/',
            'plot_title': "V(t)",  # Update the plot title for the voltage plot
            'time_series_x_label': "Time\ (s)",
            'time_series_y_label': "V\ (mV)",  # Update the y-axis label to reflect voltage
            'index': 2,  # Assuming the voltage data is in the first column
            'time_indices': "all", #[10000,125000],
            'xlim': 100,
        }
    ]


    # Will_figures_(Spiking_Problems)

    # Iterate through the list of file information dictionaries
    for list_index, file_info in enumerate(file_info_list):
        if list_index not in [9,10]:
            continue
        filepath = file_info['filepath']
        filename = os.path.splitext(os.path.basename(filepath))[0]

        data = np.loadtxt(filepath)
        if file_info['index'] != None:
            data = data[:, file_info['index']]

        # Take time slice if necessary
        if file_info['time_indices'] != "all":
            data = data[file_info['time_indices'][0] : file_info['time_indices'][1]]

        # Check if the file is a CM data file
        is_cm_data_file = 'CM_data_and_prediction' in file_info['save_folder']

        # Update the save_filename based on the file type
        quantity = "Current" if "I_{stim}" in file_info['plot_title'] else "Voltage"
        if is_cm_data_file:
            save_filename = f"FPS_{quantity}_in_{filename}"
        else:
            save_filename = f"FPS_{quantity}_in_{filename}"
            # save_filename = "Power spectrum of training current from (" + filename + ")"

        # Generate and save the Fourier Power Spectrum plot
        Fourier_Power_Spectrum_plot_and_save(
            data,
            filename,
            file_info['sampling_rate'],
            file_info['save_folder'],
            r"$Fourier\ Power\ Spectrum\ of\ " + file_info['plot_title'] + "$",
            save_filename,
            file_info['xlim'],
        )

        # # Determine whether to use seconds or milliseconds for time units
        # x_unit_should_be_seconds = "(s)" in file_info['time_series_x_label']
        # time_unit_conversion_factor = 1000 * x_unit_should_be_seconds + 1 * (not x_unit_should_be_seconds)
        #
        # # Generate and save the time series plot
        # make_general_plot.time_series_plot_and_save(
        #     data,
        #     delta_t=1.0 / file_info['sampling_rate'] * time_unit_conversion_factor,  # delta_t should be in ms
        #     save_folder=file_info['save_folder'],
        #     title=r"$" + file_info['plot_title'] + "$",
        #     save_filename=filename,
        #     xlabel=r"$" + file_info['time_series_x_label'] + "$",
        #     ylabel=r"$" + file_info['time_series_y_label'] + "$",
        # )

if __name__ == '__main__':
    main()

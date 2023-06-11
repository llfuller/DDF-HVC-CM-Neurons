import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import make_general_plot

matplotlib.rc('text', usetex=True)  # sets text.usetex to True in order to use LaTeX for labels
plt.rc('font', family='serif')


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
    fourier_transform = np.fft.rfft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))

    plt.figure()
    freq_without_0_index = frequency[1:]
    normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))

    plt.fill_between(freq_without_0_index, 0, normalized_power_spec_without_0_index, color='black', alpha=0.5)
    plt.plot(freq_without_0_index, normalized_power_spec_without_0_index, color='black', linewidth=0.5)
    plt.title(title, fontsize=16)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Normalized Power (1.0 = max of spectrum indices [1:])", fontsize=14)
    plt.xlim(0, xlim)

    ax = plt.gca()
    ax.set_facecolor("white")
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', length=6, width=1, color='black', labelsize=12)

    if normalized_power_spec_without_0_index.max() >= 1e3 or normalized_power_spec_without_0_index.min() <= 1e-3:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    plt.tight_layout()

    # Save the plot and data
    plt.savefig(save_folder + save_filename + ".png")
    plt.savefig(save_folder + save_filename + ".pdf")
    np.savetxt(save_folder + save_filename + ".txt",
               np.column_stack((freq_without_0_index, normalized_power_spec_without_0_index)))
    plt.close()




# Example usage
filepaths = [
            'I_stimuli/I_colpitts_x_time_dilation=0.2range=300_(I).txt',
             'I_stimuli/I_colpitts_x_time_dilation=0.5range=300_(I).txt',
             'I_stimuli/I_L63_x_time_dilation=0.5range=300_(I).txt',
            'CM_data_and_prediction/Experimental/2014_09_10_0013_VIt.txt',
            'CM_data_and_prediction/Experimental/2014_12_11_0017_VIt.txt'
             ]

filenames = {}
plot_titles_dict = {}
for a_filepath in filepaths:
    filenames[a_filepath] = os.path.splitext(os.path.basename(a_filepath))[0]

plot_titles_dict = { # Do not use $$ notation here
    filepaths[0]: "I_{Colpitts-x};\ td=0.2",
    filepaths[1]: "I_{Colpitts-x};\ td=0.5",
    filepaths[2]: "I_{Lorenz63-x};\ td=0.5",
    filepaths[3]: "I_{stim}(t)",
    filepaths[4]: "I_{stim}(t)"
}

time_series_y_label_dict = { # Do not use $$ notation here
    filepaths[0]: "I_{stim}\ (pA)",
    filepaths[1]: "I_{stim}\ (pA)",
    filepaths[2]: "I_{stim}\ (pA)",
    filepaths[3]: "I_{stim}\ (pA)",
    filepaths[4]: "I_{stim}\ (pA)"
}

time_series_x_label_dict = { # Do not use $$ notation here
    filepaths[0]: "Time\ (s)",
    filepaths[1]: "Time\ (s)",
    filepaths[2]: "Time\ (s)",
    filepaths[3]: "Time\ (s)",
    filepaths[4]: "Time\ (s)"
}

index_dict = {}
for filepath in filepaths:
    index_dict[filepath] = None # Index to use for data
index_dict["CM_data_and_prediction/Experimental/2014_09_10_0013_VIt.txt"] = 1 # 32425a75 Phasic
index_dict["CM_data_and_prediction/Experimental/2014_12_11_0017_VIt.txt"] = 1 # 920061fe Tonic


settings = {
    filepaths[0] : {'save_folder': 'I_stimuli/', 'sampling_rate': 50000, 'name': filenames[filepaths[0]], 'title': r"$Fourier\ Power\ Spectrum\ of\ " + plot_titles_dict[filepaths[0]] + "$", 'save_filename': "Power spectrum of training current from (" + filenames[filepaths[0]] + ")", 'xlim': 30},
    filepaths[1] : {'save_folder': 'I_stimuli/', 'sampling_rate': 50000, 'name': filenames[filepaths[1]], 'title': r"$Fourier\ Power\ Spectrum\ of\ " + plot_titles_dict[filepaths[1]] + "$", 'save_filename': "Power spectrum of training current from (" + filenames[filepaths[1]] + ")", 'xlim': 30},
    filepaths[2] : {'save_folder': 'I_stimuli/', 'sampling_rate': 50000, 'name': filenames[filepaths[2]], 'title': r"$Fourier\ Power\ Spectrum\ of\ " + plot_titles_dict[filepaths[2]] + "$", 'save_filename': "Power spectrum of training current from (" + filenames[filepaths[2]] + ")", 'xlim': 30},
    filepaths[3] : {'save_folder': os.path.dirname(filepaths[3])+"/", 'sampling_rate': 40000, 'name': "2014_09_10_0013_I", 'title': r"$Fourier\ Power\ Spectrum\ of\ " + plot_titles_dict[filepaths[3]] + "$", 'save_filename': "FPS_Current_in_" + filenames[filepaths[3]], 'xlim': 60},
    filepaths[4] : {'save_folder': os.path.dirname(filepaths[4])+"/", 'sampling_rate': 40000, 'name': "2014_12_11_0017_I", 'title': r"$Fourier\ Power\ Spectrum\ of\ " + plot_titles_dict[filepaths[4]] + "$", 'save_filename': "FPS_Current_in_" + filenames[filepaths[4]], 'xlim': 60},
}

for filepath in filepaths[:]:
    data = np.loadtxt(filepath)
    if index_dict[filepath] != None:
        data = data[:,index_dict[filepath]]
    Fourier_Power_Spectrum_plot_and_save(data, settings[filepath]['name'], settings[filepath]['sampling_rate'], settings[filepath]['save_folder'], settings[filepath]['title'], settings[filepath]['save_filename'],  settings[filepath]['xlim'])
    if ".atf" in filepath:
        np.savetxt(os.path.splitext(filepath)[0]+".txt", data.T) # create text file of .atf file
    x_unit_should_be_seconds = ("(s)" in time_series_x_label_dict[filepath])
    time_unit_conversion_factor = (1000*x_unit_should_be_seconds + 1*(not x_unit_should_be_seconds))
    make_general_plot.time_series_plot_and_save(data, delta_t=1.0/settings[filepath]['sampling_rate']*time_unit_conversion_factor, #delta_t should be is ms
                                                save_folder=settings[filepath]['save_folder'],
                                                title=r"$"+plot_titles_dict[filepath]+"$",
                                                save_filename=settings[filepath]['name'],
                                                xlabel=r"$"+time_series_x_label_dict[filepath]+"$", ylabel=r"$"+time_series_y_label_dict[filepath]+"$")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytinytex

matplotlib.rc('text', usetex=True) # sets text.usetex to True in order to use LaTeX for labels
plt.rc('font', family='serif')

# Make plots from Eric

# Function to read data from file
def read_data(file_path):
    data = np.loadtxt(file_path)
    time = data[:, 0]   #  time already in milliseconds
    true_voltage = data[:, 1]  #  (mV)
    predicted_voltage = data[:, 2]  # (mV)
    return time, true_voltage, predicted_voltage

# List of file names
file_names = [
    "prediction_Red171_Neuron_1_500_centers_poor_subthreshold_1.txt",
    "prediction_Red171_Neuron_1_500_centers_good_subthreshold_1.txt",
    "prediction_Red171_Neuron_1_(500+1000)_Centers.txt"
]

# Use 'default' style
plt.style.use('default')

for file_name in file_names:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Read data from file
    time, true_voltage, predicted_voltage = read_data(file_name)

    time_in_seconds = time/1000.0

    # Plot true voltage (solid black line)
    ax.plot(time_in_seconds, true_voltage, color='black', label='True Voltage')

    # Plot predicted voltage (dashed red line)
    ax.plot(time_in_seconds, predicted_voltage, color='red', linestyle='--', label='Prediction')

    # Set axes labels with bold font and increased font size
    ax.set_xlabel(r'$Time\ (s)$', fontweight='bold', fontsize=14)
    ax.set_ylabel(r'$Voltage\ (mV)$', fontweight='bold', fontsize=14)

    # Add gridlines
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)

    # Add legend
    ax.legend()

    # Set title based on the file name
    if "500_centers" in file_name:
        ax.set_title(r"$500\ Centers$")
    elif "(500+1000)_Centers" in file_name:
        ax.set_title(r"$1500\ Centers$")

    # Save the plot as a separate .png file
    output_filename = file_name.replace('.txt', '.pdf')
    plt.savefig(output_filename, dpi=300)
    output_filename = file_name.replace('.txt', '.png')
    plt.savefig(output_filename, dpi=300)


    # Close the current figure to release memory
    plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np

# Make Will's plots (originally Figures 8-11)

# Function to read data from file
def read_data(file_path):
    data = np.loadtxt(file_path)
    time = data[:, 0] * 1000  # Convert time from seconds to milliseconds
    predicted_voltage = data[:, 1]
    true_voltage = data[:, 2]
    return time, predicted_voltage, true_voltage

# List of file names
file_names = [
    "Bad Kansa 1.txt", "Bad Kansa 2.txt", "Good Kansa 1.txt", "Good Kansa 2.txt",
    "Naive Adaptive 1.txt", "Naive Adaptive 2.txt", "Long and Short 1.txt", "Long and Short 2.txt"
]

# Use 'default' style
plt.style.use('default')

for file_name in file_names:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Read data from file
    time, predicted_voltage, true_voltage = read_data(file_name)

    # Plot true voltage (solid black line)
    ax.plot(time, true_voltage, color='black', label='True Voltage')

    # Plot predicted voltage (dashed red line)
    ax.plot(time, predicted_voltage, color='red', linestyle='--', label='Prediction')

    # Set axes labels with bold font and increased font size
    ax.set_xlabel('Time (ms)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Voltage (mV)', fontweight='bold', fontsize=14)

    # Add gridlines
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)

    # Add legend
    ax.legend()

    # Set title (leave it blank)
    ax.set_title('Neuron 52')

    # Save the plot as a separate .png file
    output_filename = file_name.replace('.txt', '.png')
    plt.savefig(output_filename, dpi=300)

    # Close the current figure to release memory
    plt.close(fig)

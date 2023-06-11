import numpy as np
import matplotlib.pyplot as plt

# Set the properties for using LaTeX and a serif font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the data from the text files
time = np.loadtxt('time.txt')
voltage_prediction = np.loadtxt('voltage_prediction.txt')
voltage_truth = np.loadtxt('voltage_truth.txt')

# Create the plot with the specified formatting
plt.figure()
plt.plot(time, voltage_prediction, 'r--', label='Voltage Prediction (mV)')
plt.plot(time, voltage_truth, 'k', label='Voltage Truth (mV)')

# Add labels, title, and gridlines
plt.xlabel(r'Time (s)', fontsize=14)
plt.ylabel(r'Voltage (mV)', fontsize=14)
plt.title(r'\textbf{Placeholder Title}', fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.grid(which='both', linestyle=':', color='lightgrey')

# Display the plot
plt.show()

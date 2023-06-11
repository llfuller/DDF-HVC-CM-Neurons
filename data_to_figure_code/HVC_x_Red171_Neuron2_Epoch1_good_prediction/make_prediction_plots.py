import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytinytex
import os

matplotlib.rc('text', usetex=True) # sets text.usetex to True in order to use LaTeX for labels
plt.rc('font', family='serif')

# Set the properties for using LaTeX and a serif font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#==============================================32425a75 Figure (Phasic)==========================================================
# Load the data from the text files
time_1 = np.loadtxt("_epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_time.txt")   #  time already in milliseconds
voltage_prediction_1 = np.loadtxt("_epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_voltage_truth.txt")  #  (mV)
voltage_truth_1 = np.loadtxt("_epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_voltage_prediction.txt")  # (mV)

# Create the plot with the specified formatting
plt.figure()
plt.plot(time_1, voltage_truth_1, 'k', label='Voltage Truth (mV)')
plt.plot(time_1, voltage_prediction_1, 'r--', label='Voltage Prediction (mV)', linewidth= 0.5)

# Add labels, title, and gridlines
plt.xlabel(r'$Time\ (s)$', fontsize=14)
plt.ylabel(r'$Voltage\ (mV)$', fontsize=14)
plt.title(r"$DDF\ HVC_{RA}\ Prediction$", fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='lightgrey')

# Display the plot
plt.show()
# Save the plot as a separate .png file
output_filename = "epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_voltage_prediction.pdf"
plt.savefig(output_filename, dpi=300)
output_filename = "epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_voltage_prediction.png"
plt.savefig(output_filename, dpi=300)


# Close the current figure to release memory
plt.close()
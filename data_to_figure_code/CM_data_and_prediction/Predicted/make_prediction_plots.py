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
time_1 = np.loadtxt("32425a75_2014_09_10_0013 with tstep=2.5e-05 s, D = 6, Beta = 1.0e+01, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 3_time.txt")   #  time already in milliseconds
voltage_prediction_1 = np.loadtxt("32425a75_2014_09_10_0013 with tstep=2.5e-05 s, D = 6, Beta = 1.0e+01, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 3_voltage_truth.txt")  #  (mV)
voltage_truth_1 = np.loadtxt("32425a75_2014_09_10_0013 with tstep=2.5e-05 s, D = 6, Beta = 1.0e+01, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 3_voltage_prediction.txt")  # (mV)

# Create the plot with the specified formatting
plt.figure()
plt.plot(time_1, voltage_truth_1, 'k', label='Voltage Truth (mV)')
plt.plot(time_1, voltage_prediction_1, 'r--', label='Voltage Prediction (mV)', linewidth= 0.5)

# Add labels, title, and gridlines
plt.xlabel(r'$Time\ (s)$', fontsize=14)
plt.ylabel(r'$Voltage\ (mV)$', fontsize=14)
plt.title(r"$DDF\ Phasic\ CM\ Prediction$", fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='lightgrey')

# Display the plot
plt.show()
# Save the plot as a separate .png file
output_filename = "32425a75_2014_09_10_0013 with tstep=2.5e-05 s, D = 6, Beta = 1.0e+01, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 3_voltage_prediction.pdf"
plt.savefig(output_filename, dpi=300)
output_filename = "32425a75_2014_09_10_0013 with tstep=2.5e-05 s, D = 6, Beta = 1.0e+01, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 3_voltage_prediction.png"
plt.savefig(output_filename, dpi=300)


# Close the current figure to release memory
plt.close()

#==============================================920061fe Figure (Tonic)==========================================================
# Load the data from the text files
time_2 = np.loadtxt("920061fe_2014_12_11_0017 with tstep=2.5e-05 s, D = 4, Beta = 1.0e+00, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 4_time.txt")
voltage_prediction_2 = np.loadtxt("920061fe_2014_12_11_0017 with tstep=2.5e-05 s, D = 4, Beta = 1.0e+00, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 4_voltage_truth.txt")
voltage_truth_2 = np.loadtxt("920061fe_2014_12_11_0017 with tstep=2.5e-05 s, D = 4, Beta = 1.0e+00, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 4_voltage_prediction.txt")

# Create the plot with the specified formatting
plt.figure()
plt.plot(time_2, voltage_truth_2, 'k', label='Voltage Truth (mV)')
plt.plot(time_2, voltage_prediction_2, 'r--', label='Voltage Prediction (mV)', linewidth= 0.5)

# Add labels, title, and gridlines
plt.xlabel(r'$Time\ (s)$', fontsize=14)
plt.ylabel(r'$Voltage\ (mV)$', fontsize=14)
plt.title(r"$DDF\ Tonic\ CM\ Prediction$", fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='lightgrey')

# Display the plot
plt.show()

# Save the plot as a separate .png file
output_filename = "920061fe_2014_12_11_0017 with tstep=2.5e-05 s, D = 4, Beta = 1.0e+00, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 4_voltage_prediction.pdf"
plt.savefig(output_filename, dpi=300)
output_filename = "920061fe_2014_12_11_0017 with tstep=2.5e-05 s, D = 4, Beta = 1.0e+00, R = 1.0e-02 Train TSteps = 399000, Centers = 500, tau = 4_voltage_prediction.png"
plt.savefig(output_filename, dpi=300)

# Close the current figure to release memory
plt.close()
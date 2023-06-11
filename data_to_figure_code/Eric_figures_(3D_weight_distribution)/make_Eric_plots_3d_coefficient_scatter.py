import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pytinytex

matplotlib.rc('text', usetex=True) # sets text.usetex to True in order to use LaTeX for labels
plt.rc('font', family='serif')

def plot_time_delay_center_values(file_name, columns_tuple=(0,1,2)):
    # Read data from the file
    if "(500+1000)" in file_name:
        data = np.loadtxt(file_name, delimiter=' ', skiprows=1)
        columns_tuple = (0, 4, 9)
    else:
        data = np.loadtxt(file_name, delimiter='\t', skiprows=1)

    # Extract columns
    v_t = data[:, columns_tuple[0]]
    v_t_5tau = data[:, columns_tuple[1]]
    v_t_10tau = data[:, columns_tuple[2]]
    center_weight_coeff = data[:, 3]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the colormap
    cmap = plt.get_cmap('bwr')

    # Plot the data points with the center_weight_coeff values for coloring
    sc = ax.scatter(v_t, v_t_5tau, v_t_10tau, c=center_weight_coeff, cmap=cmap, edgecolors='black', linewidths=0.13, s=15, alpha=0.7)

    # Set labels for the axes with LaTeX formatting
    ax.set_xlabel(r'$V(t) \, (\mathrm{mV})$', fontweight='bold', fontsize=14)
    ax.set_ylabel(r'$V(t-5\tau) \, (\mathrm{mV})$', fontweight='bold', fontsize=14)
    ax.set_zlabel(r'$V(t-10\tau) \, (\mathrm{mV})$', fontweight='bold', fontsize=14)

    # Set the title
    ax.set_title('$Time\ Delay\ Center\ Values$')

    # Adjust the plot layout to make it tighter
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

    # Add a colorbar
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    cbar = plt.colorbar(sc, cax=cbar_ax, cmap=cmap, pad=0.1)
    cbar.set_label(r'Center Coefficient Values')

    # Save the plot using the same name as the input text file, with a .png extension
    output_file_name = file_name.replace('.txt', '.pdf')
    plt.savefig(output_file_name, bbox_inches='tight')
    output_file_name = file_name.replace('.txt', '.png')
    plt.savefig(output_file_name, bbox_inches='tight')

    # Close the plot to prevent it from being displayed
    plt.close(fig)

# Call the function with your text files
plot_time_delay_center_values("3D_Centers(500)_Red171_Neuron1_Epoch1_good_subthreshold_1.txt")
plot_time_delay_center_values("3D_Centers(500)_Red171_Neuron1_Epoch1_poor_subthreshold_1.txt")
plot_time_delay_center_values("3D_Centers(500+1000)_Red171_Neuron1_Epoch1_more_spread_out.txt")
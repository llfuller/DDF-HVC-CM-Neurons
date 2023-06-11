import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

def plot_time_delay_center_values(file_name):
    # Read data from the file
    data = np.loadtxt(file_name, delimiter='\t', skiprows=1)

    # Extract columns
    v_t = data[:, 0]
    v_t_5tau = data[:, 1]
    v_t_10tau = data[:, 2]
    center_weight_coeff = data[:, 3]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the colormap
    cmap = plt.get_cmap('bwr')

    # Normalize the center weight coefficient for colormap
    norm = plt.Normalize(center_weight_coeff.min(), center_weight_coeff.max())
    colors = cmap(norm(center_weight_coeff))

    # Gaussian marker function
    def plot_gaussian_marker(ax, x, y, z, color, marker_size):
        N = 10
        xy = np.mgrid[-1:1:1j * N, -1:1:1j * N].reshape(2, -1).T
        z_offset = multivariate_normal.pdf(xy, mean=[0, 0], cov=[[0.5, 0], [0, 0.5]])
        z_offset /= z_offset.max()

        # Calculate new coordinates with vectorized operations
        x_coords = x + xy[:, 0]
        y_coords = y + xy[:, 1]
        z_coords = z + z_offset

        # Plot all points in one call
        ax.scatter(x_coords, y_coords, z_coords, c=color, marker='.', alpha=0.1, s=marker_size)

    # Plot the data points with Gaussian coloration
    marker_size = 0.01
    for i in range(len(v_t)):
        plot_gaussian_marker(ax, v_t[i], v_t_5tau[i], v_t_10tau[i], colors[i], marker_size)

    # Set labels for the axes with LaTeX formatting
    ax.set_xlabel(r'$V(t) \, (\mathrm{mV})$', fontweight='bold', fontsize=14)
    ax.set_ylabel(r'$V(t-5\tau) \, (\mathrm{mV})$', fontweight='bold', fontsize=14)
    ax.set_zlabel(r'$V(t-10\tau) \, (\mathrm{mV})$', fontweight='bold', fontsize=14)

    # Set the title
    ax.set_title('Time Delay Center Values')

    # Adjust the plot layout to make it tighter
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

    # Add a colorbar
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, pad=0.1)
    cbar.set_label('Center Weight Coefficient')

    # Save the plot using the same name as the input text file, with a .png extension
    output_file_name = file_name.replace('.txt', '.pdf')
    plt.savefig(output_file_name, bbox_inches='tight')

    # Close the plot to prevent it from being displayed
    plt.close(fig)


# Call the function with your text files
plot_time_delay_center_values("figure_16_data.txt")
plot_time_delay_center_values("figure_18_data.txt")
plot_time_delay_center_values("figure_20_data.txt")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytinytex

matplotlib.rc('text', usetex=True) # sets text.usetex to True in order to use LaTeX for labels
plt.rc('font', family='serif')

# Creates figure 23 and 24 in the original paper draft sent to Henry (FNN ratio vs D_E for given R value)

def plot_and_save(filename):
    # Load the data from the text file
    data = np.loadtxt(filename)

    # Separate the data into columns
    D_E = data[:, 0]
    FNN_ratio = data[:, 1]

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(D_E, FNN_ratio, marker='o', linestyle='-', color='black')

    # Set the x-axis to integers only
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add major gridlines
    ax.grid(True)

    # Add minor gridlines with lighter grey color and custom spacing
    ax.set_xticks(np.arange(min(D_E), max(D_E)+1, 1), minor=True)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='lightgrey')

    # Set the title to the filename with underscores replaced by spaces
    filename_without_extension = filename.replace(".txt", "")

    title = ''
    if "FNN_vs_D_R=0.0001,window=1000" == filename_without_extension:
        title = r"$FNN (r_{FNN}=0.0001,\ window=1000\ timesteps)$"
    if "FNN_vs_D_R=0.0001,window=100000" == filename_without_extension:
        title = r"$FNN (r_{FNN}=0.0001,\ window=100000\ timesteps)$"

    ax.set_title(title)

    # Set x-axis and y-axis labels in LaTeX format
    ax.set_xlabel(r'$D_E$', fontsize=12)
    ax.set_ylabel(r'$FNN\ ratio$', fontsize=12)

    # Remove unnecessary white space around the plot
    plt.tight_layout()

    # Save the plot as a .png file
    plt.savefig(filename.replace('.txt', '.png'), dpi=300)
    plt.savefig(filename.replace('.txt', '.pdf'))

    # Show the plot
    plt.show()

# Call the function for each file
file1 = "FNN_vs_D_R=0.0001,window=1000.txt"
file2 = "FNN_vs_D_R=0.0001,window=100000.txt"

plot_and_save(file1)
plot_and_save(file2)
import matplotlib.pyplot as plt
import save_utilities

def plotting_quantity(x_arr, y_arr, title,xlabel,ylabel,save_and_or_display="save",save_location=None,xlim=None):
    """
    A simple Matplotlib.pyplot plotting function to save space. Will save and/or display if specified
    returns: Matplotlib.pyplot figure object
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_arr, y_arr)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if "current" in title.lower():
        ax.get_lines()[0].set_color("orange")
    if "voltage" in title.lower():
        ax.get_lines()[0].set_color("blue")
    plt.xlim(xlim)
    save_utilities.save_and_or_display_plot(fig, save_and_or_display, save_location)
    return fig

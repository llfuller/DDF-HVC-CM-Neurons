import matplotlib.pyplot as plt
import numpy as np
import os.path

def save_fig_with_makedir(figure, save_location):
    """
    input: matplotlib figure fig, and string "save_location" (example: directory1/directory2/..../filename.ext)
    Creates all directories within save_location if they don't yet exist, then saves figure fig to save_location
    """
    if "/" in save_location:
        last_slash_index = save_location.rfind('/') #finds last location of "/" in save_location
    directory = save_location[:last_slash_index]
    filename  = save_location[last_slash_index:]
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if os.path.isdir(directory):
        figure.savefig(directory + str(filename), bbox_inches='tight')

def save_txt_with_makedir(data, save_location):
    """
    input: text data, and string "save_location" (example: directory1/directory2/..../filename.ext)
    Creates all directories within save_location if they don't yet exist, then saves data to save_location
    """
    if "/" in save_location:
        last_slash_index = save_location.rfind('/') #finds last location of "/" in save_location
    directory = save_location[:last_slash_index]
    filename  = save_location[last_slash_index:]
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if os.path.isdir(directory):
        np.savetxt(directory + str(filename), data)


def save_and_or_display_plot(figure, a_str, save_location):
    if "save" in a_str.lower():
        save_fig_with_makedir(figure,save_location)
    if "display" not in a_str.lower():
        figure.close()

def save_text(data, a_str, save_location):
    if "save" in a_str.lower():
        save_txt_with_makedir(data, save_location)
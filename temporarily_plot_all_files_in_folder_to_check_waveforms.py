# Purpose: Iterates through given folder .txt column files V(t) and I(t) (separate files) and save figures in a folder
#          Used to evaluate signals quickly
# Author: Lawson Fuller
# Date: June 4, 2022

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
directories_list = ["HVC_ra_x_i_data_2015", #Arij Daou 2015 HVC data
                    "HVC_ra_x_i_data_2016_2019/08-15-2019"] #Arij Daou 2016 HVC data
saving_directory = "experimental_data_plots"

# Do not modify below
for a_directory in directories_list:
    for subdir, dirs, files in os.walk(a_directory,topdown="False"):
        for file in files:
            filed_extension = file[-4:]
            if (filed_extension==".txt"):
                path_plus_filename = os.path.join(subdir, file) # includes loaded file's extension
                filename_without_extension = file[:-4]#Path(path_plus_filename).stem
                dir_to_store_figs = saving_directory+"/"+str(subdir)+"/"
                data=np.loadtxt(path_plus_filename)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(data)
                if not os.path.isdir(dir_to_store_figs):
                    os.makedirs(dir_to_store_figs)
                if os.path.isdir(dir_to_store_figs):
                    print("saving:\n"+str(dir_to_store_figs+str(filename_without_extension)+".png"))
                    fig.savefig(dir_to_store_figs+str(filename_without_extension)+".png", bbox_inches='tight')
                    print("-----------------------------------")
                plt.close(fig)
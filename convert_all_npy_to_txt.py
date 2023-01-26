import numpy as np
import matplotlib.pyplot as plt
import save_utilities
import glob

root_directory = "converted_npy_to_txt/"
file_extension = "npy"
# ======== do not modify below ==========
full_paths_list = glob.glob(root_directory+"**/*location."+str(save_utilities.glob_extension_case_string_builder(file_extension)),
                            recursive=True)
for i, path in enumerate(full_paths_list):
    full_paths_list[i] = path.replace("\\","/")

extensions_with_included_unit_data = []

print("Full paths list:"+str(full_paths_list))

for a_path in full_paths_list:
    last_slash_location = a_path.rfind("/")
    a_filename = a_path[last_slash_location+1:]
    print(f"Using file: {a_filename}")
    directory_to_read_input_data = a_path[:last_slash_location+1] # should include the last slash, but nothing past it
    directory_to_store_plots = "converted_npy_to_txt/plots/"
    directory_to_store_txt_data = "converted_npy_to_txt/data_derived/"
    imported_npy = np.load(a_path)
    save_utilities.save_text(data=imported_npy, a_str="save", save_location=directory_to_store_txt_data+a_filename[:-4]+'.txt')
    # plt.figure()
    # plt.plot(imported_npy)
    # plt.show()
import os
import difflib
import platform
import subprocess

# Using filepath and search directory, search the search directory for file with name most similar to the name in filepath

def get_closest_match(filename, directory):
    files = os.listdir(directory)
    match = difflib.get_close_matches(filename, files, n=1, cutoff=0.0)
    return match[0] if match else None

def open_directory_with_selection(directory, closest_match):
    if platform.system() == "Windows":
        # Windows
        os.startfile(os.path.join(directory, closest_match))
    elif platform.system() == "Darwin":
        # macOS
        subprocess.Popen(["open", "-R", os.path.join(directory, closest_match)])
    else:
        # Linux
        subprocess.Popen(["xdg-open", os.path.join(directory, closest_match)])

def main(filestring, search_dir):
    if not os.path.exists(filestring):
        print("File not found.")
        return

    filename = os.path.basename(filestring)

    if not os.path.exists(search_dir):
        print("Search directory not found.")
        return

    closest_match = get_closest_match(filename, search_dir)
    if closest_match:
        print(f"The closest match is: {closest_match}")
        open_directory_with_selection(search_dir, closest_match)
    else:
        print("No matching file found.")

if __name__ == "__main__":
    filestring = "_epoch_1 with tstep=0.02 ms, D = 10, Beta = 1.0e-03, R = 1.0e-03 Train TSteps = 499000, Centers = 500, tau = 10_Convolved_waveforms_sigma=5"
    search_dir = ""
    main(filestring, search_dir)

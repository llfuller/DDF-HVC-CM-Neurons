import os
from pathlib import Path
import quantities as pq
from neo.io import AxonIO
import numpy as np

def create_readme(output_dir):
    """
    Create a README file with relevant meta information about the files.

    Args:
        output_dir (Path): The directory where the README file will be created.
    """
    readme_path = output_dir / 'README.txt'
    with open(readme_path, 'w') as readme:
        readme.write("README\n")
        readme.write("------\n\n")
        readme.write("This directory's .abf files are from Meliza, but the .txt files are not.\n\n")
        readme.write("Data Structure:\n")
        readme.write("- Each .abf file has its own directory under 'txt_V_I_t' subdirectory within the original .abf file's parent directory.\n")
        readme.write("- Within the .abf file directory, each epoch has its own directory, named 'epoch_X', where X is the epoch number.\n")
        readme.write("- Within each epoch directory, there are text files for each segment (sweep) of the epoch.\n")
        readme.write("- The segment files are named 'epoch_X_segment_Y.txt', where X is the epoch number and Y is the segment number.\n\n")
        readme.write("File Format:\n")
        readme.write("- Each segment file contains three columns: Voltage (V), Current (I), and Time (s).\n")
        readme.write("Dimensionality Files:\n")
        readme.write("- For each segment file, there is a corresponding dimensionality file named 'dimensionality_epoch_X_segment_Y.txt', where X is the epoch number and Y is the segment number.\n")
        readme.write("- These files contain the dimensionality of the Voltage, Current, and Time columns, separated by spaces.\n")



def find_abf_files(dir_path):
    """
    Recursively find all .abf files in a given directory.

    Args:
        dir_path (str): The path to the directory to search in.

    Returns:
        list: The list of .abf file paths.
    """
    abf_files = []
    for entry in os.scandir(dir_path):
        if entry.is_file() and entry.path.endswith('.abf'):
            abf_files.append(entry.path)
        elif entry.is_dir():
            abf_files.extend(find_abf_files(entry.path))
    return abf_files


def extract_segment_data(segment):
    """
    Extract data and dimensionality from a segment in a block.

    Args:
        segment (Segment): A segment from a Neo block.

    Returns:
        tuple: A tuple containing Time, Voltage, and Current arrays and their dimensionalities.
    """
    junction_potential = pq.Quantity(11.6, 'mV')  # measured at 32 C
    V = segment.analogsignals[0] - junction_potential
    I = segment.analogsignals[1]
    T = V.times - V.t_start

    return T, V, I, T.dimensionality, V.dimensionality, I.dimensionality

def save_data(output_dir, file_name, T, V, I):
    """
    Save Time, Voltage, and Current data to a .txt file.

    Args:
        output_dir (Path): The directory to save the .txt file in.
        file_name (str): The name of the .txt file.
        T (array): Time data.
        V (array): Voltage data.
        I (array): Current data.
    """
    data_matrix = np.column_stack((V.magnitude, I.magnitude, T))
    file_path = output_dir / file_name
    np.savetxt(file_path, data_matrix, delimiter=' ', fmt='%.18e')


def save_dimensionality(output_dir, file_name, t_dim, v_dim, i_dim):
    """
    Save dimensionality data to a .txt file.

    Args:
        output_dir (Path): The directory to save the .txt file in.
        file_name (Path): The name of the .txt file.
        t_dim (str): Time dimensionality.
        v_dim (str): Voltage dimensionality.
        i_dim (str): Current dimensionality.
    """
    with open(output_dir / (file_name), 'w') as dim_file:
        dim_file.write(f"{t_dim} {v_dim} {i_dim}\n")



def main():
    # Set the root directory and create the output directory
    root_dir = Path('./')
    search_dir = root_dir / 'cm_ddf'
    output_dir = root_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .abf files in the search directory
    abf_files = find_abf_files(search_dir)

    # Extract data from each .abf file and save it as .txt files
    for abf_path in abf_files:
        abf_path = Path(abf_path)
        relative_path = 'cm_ddf' / abf_path.relative_to(search_dir)

        # Extract the epoch number from the .abf file name
        epoch_number = int(abf_path.stem.split('_')[-1])

        # Read the block from the .abf file
        fp = AxonIO(filename=abf_path)
        block = fp.read_block()

        # Iterate through each segment and save the data and dimensionality
        for i, segment in enumerate(block.segments):
            T, V, I, t_dim, v_dim, i_dim = extract_segment_data(segment)

            segment_output_dir = output_dir / relative_path.parent / 'txt_V_I_t' / abf_path.stem / f'epoch_{epoch_number}'
            segment_output_dir.mkdir(parents=True, exist_ok=True)

            save_data(segment_output_dir, f'epoch_{epoch_number}_segment_{i}.txt', T, V, I)
            save_dimensionality(segment_output_dir, f'dimensionality_epoch_{epoch_number}_segment_{i}.txt', t_dim, v_dim, i_dim)

    # Create a README file with meta information
    create_readme(output_dir / 'cm_ddf')

if __name__ == '__main__':
    main()

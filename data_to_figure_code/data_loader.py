import numpy as np
import json
import copy
import sys
import os
import re
sys.path.append("./data_to_figure_code")
from make_general_plot import time_series_plot_and_save
import make_time_delay_embedding
import matplotlib.pyplot as plt
##================== Text file methods =======================
import numpy as np


def load_datasets_from_txt(filepath, column_name_num_dict):
    """
    Load datasets from a text file.

    Parameters
    ----------
    filepath (str): Full path to the file from the working directory.
    column_name_num_dict (dict): Dictionary with key as the physical quantity and value as the column index.

    Returns
    -------
    dict: A dictionary containing the loaded data arrays for each physical quantity specified in column_name_num_dict.
    """
    if ".txt" in filepath:
        data = np.loadtxt(filepath)
        # Initialize quantity arrays inside dict
        return_dict = {}

        if len(data.shape) == 1:  # If there is only one row in the file
            for quantity, column_index in column_name_num_dict.items():
                if column_index < len(data):
                    return_dict[quantity] = data[column_index]
                else:
                    raise ValueError(f"Column index {column_index} for quantity '{quantity}' is out of range")
        else:  # If there are multiple rows in the file
            for quantity, column_index in column_name_num_dict.items():
                return_dict[quantity] = data[:, column_index]

        return return_dict


def collect_txt_files(directory, required_substrings, excluded_substrings):
    """
    Recursively scans the given directory and returns a list of .txt files
    that contain all required_substrings and do not contain any of the excluded_substrings.

    :param directory: str, the path of the directory to scan
    :param required_substrings: list, a list of required substrings in the filename
    :param excluded_substrings: list, a list of excluded substrings in the filename
    :return: list, a list of file paths
    """
    # file_list = []
    #
    # for entry in os.scandir(directory):
    #     if entry.is_file() and entry.name.endswith('.txt'):
    #         # Check if the file contains all required_substrings and none of the excluded_substrings
    #         if all(substr in entry.name for substr in required_substrings) and \
    #                 not any(substr in entry.name for substr in excluded_substrings):
    #             # Append the file path to the file_list
    #             file_list.append(entry.path)
    #     elif entry.is_dir():
    #         # If the entry is a directory, call the function recursively and extend the file_list
    #         file_list.extend(collect_txt_files(entry.path, required_substrings, excluded_substrings))
    #
    # return file_list
    list_of_files = []
    list_of_files.clear()
    for (dirpath, dirnames, filenames) in os.walk(directory):
        print(f"Exploring directory: {dirpath}")  # Add this line to print the directory being explored
        for filename in filenames:
            if all(substr in filename for substr in required_substrings) and not any(substr in filename for substr in excluded_substrings):
                list_of_files.append(os.path.join(dirpath, filename))
    return list_of_files


# # Set the target directory, required and excluded substrings
# directory = "stimuli_April_2023/"
# required_substrings = [".txt", "(-33.33,100)"]
# excluded_substrings = ["FPS", "readme", "README"]
#
# # Call the function to collect file paths and store the result in list_of_all_stimulus_files
# list_of_all_stimulus_files = collect_txt_files(directory, required_substrings, excluded_substrings)
#
# # Print the list of file paths
# print(list_of_all_stimulus_files)

def pull_time_dilation_from_string(a_str):
    # Use a regular expression to search for the number after "time_dilation="
    match = re.search(r'time_dilation=([\d.]+)', a_str)

    # If a match is found, extract the number as a string
    if match:
        I_time_dilation = match.group(1)
        print("I_time_dilation:", I_time_dilation)
    else:
        print("No match found")

    return I_time_dilation


##================== JSON load/update methods =======================
def update_json_file(file_path, new_dicts):
    """
    Update or add dictionaries in a JSON file based on their "name" attribute.

    Args:
        file_path (str): The path to the JSON file to be updated.
        new_dicts (list): A list of dictionaries to be added or used to update existing dictionaries.

    This function reads the existing content from the specified JSON file, updates or adds dictionaries
    based on their "name" attribute, and writes the modified content back to the file.
    """
    # # Example usage:
    # my_dict_1 = {"name": "example_1", "value": "Hello, World!"}
    # my_dict_2 = {"name": "example_2", "value": "This is a test!"}
    # new_dicts = [my_dict_1, my_dict_2]
    # file_path = "output.json"
    #
    # update_json_file(file_path, new_dicts)
    try:
        with open(file_path, 'r') as file:
            existing_content = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_content = []

    for new_dict in new_dicts:
        updated = False
        for idx, existing_dict in enumerate(existing_content):
            if existing_dict.get("name") == new_dict.get("name"):
                existing_content[idx] = new_dict
                updated = True
                break

        if not updated:
            existing_content.append(new_dict)

    with open(file_path, 'w') as file:
        json.dump(existing_content, file, indent=4)




def load_datasets_from_JSON(file_path):
    """
    Load dictionaries from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries read from the JSON file. If the file does not exist or is not
              a valid JSON file, an empty list is returned.
    """
    # # Example usage:
    # file_path = "output.json"
    # loaded_dicts = load_datasets_from_JSON(file_path)
    # print(loaded_dicts)

    try:
        with open(file_path, 'r') as file:
            content = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        content = []

    return content

def load_selected_json_dicts(file_path, names_to_select):
    """
    Load selected dictionaries from a JSON file based on a list of "name" values.

    Args:
        file_path (str): The path to the JSON file.
        names_to_select (list): A list of "name" values to filter dictionaries.

    Returns:
        list: A list of dictionaries that have their "name" attribute in the provided `names_to_select` list.
              If the file does not exist or is not a valid JSON file, an empty list is returned.
    """
    # # Example usage:
    # file_path = "output.json"
    # names_to_select = ["example_1"]
    # selected_dicts = load_selected_json_dicts(file_path, names_to_select)
    # print(selected_dicts)
    try:
        with open(file_path, 'r') as file:
            content = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        content = []

    selected_dicts = []
    for name in names_to_select:
        for d in content:
            if d.get("name") == name:
                selected_dicts.append(d)
                break

    return selected_dicts


def load_txt_array_into_dicts(list_of_dataset_dicts):
    """
    Load text files into dataset dictionaries if they don't already have one in their ["V"]["array"] entry, for example.

    This function iterates over the input list of dataset dictionaries, and for each dictionary, it iterates over the
    "data_filepath_dict" key-value pairs. It loads the data from the specified text file using numpy.loadtxt() and
    assigns the corresponding arrays to the appropriate entries in the dictionary if they are not already loaded.

    Parameters
    ----------
    list_of_dataset_dicts : list of dict
        A list of dataset dictionaries like the kind outputted by load_selected_json_dicts or load_datasets_from_JSON.

    Returns
    -------
    list_of_dataset_dicts : list of dict
        The same input list of dataset dictionaries but with the arrays added into the appropriate entries.
    """

    for dataset_dict in list_of_dataset_dicts:
        data_filepath_dict = dataset_dict["data_filepath_dict"]

        for filepath, data_keys in data_filepath_dict.items():
            # Load the data from the text file
            data = np.loadtxt(filepath)

            for index, data_key in enumerate(data_keys):
                # Check if the array is not already loaded
                if dataset_dict[data_key]["array"] is None:
                    # Load the corresponding array from the data
                    dataset_dict[data_key]["array"] = data[:, index]

    return list_of_dataset_dicts


##================== Dataset methods =======================

def create_dataset_dict(name, V, I_stim, t, dt, neuron_type, collection_year, data_filepath_dict, data_filepath_original, V_units="mV", I_stim_units="pA", t_units="s", I_stim_name=None, epoch=None, extra_notes="", load_data=False, save_time_array=False):
    """
    Create a dictionary representing a dataset.

    Args:
        name (str): The dataset name.
        V (np.array): The voltage array.
        I_stim (np.array): The stimulation current array.
        t (np.array): The time array.
        dt (float): The time step.
        neuron_type (str): The type of neuron.
        collection_year (int): The year of data collection.
        data_filepath_dict (dict): A dictionary containing file paths and associated physical quantities.
        data_filepath_original (str): The original file path of the dataset.
        V_units (str, optional): Units of voltage. Defaults to "mV".
        I_stim_units (str, optional): Units of stimulation current. Defaults to "pA".
        t_units (str, optional): Units of time. Defaults to "s".
        I_stim_name (str, optional): The name of the stimulation current.
        epoch (int, optional): The epoch number.
        extra_notes (str, optional): Any additional notes. Defaults to an empty string.
        load_data (bool, optional): If True, load data from the specified file paths. Defaults to False.

    Returns:
        dict: A dictionary representing the dataset. Also creates time array if originally length 0.
    """
    dataset_dict = {
        "name": name,
        "V": {"array": V, "units": V_units},
        "I_stim": {"array": I_stim, "units": I_stim_units},
        "t": {"array": t, "units": t_units},
        "dt": dt,
        "neuron_type": neuron_type,
        "collection_year": collection_year,
        "data_filepath_dict": data_filepath_dict,
        "data_filepath_original": data_filepath_original,
        "I_stim_name" : I_stim_name,
        "epoch": epoch,
        "extra_notes": extra_notes,
    }

    if load_data:
        for filepath, quantities in data_filepath_dict.items():
            loaded_data_dict = load_datasets_from_txt(filepath, quantities)
            for quantity, data in loaded_data_dict.items():
                dataset_dict[quantity]["array"] = data
    if save_time_array:
        if len(dataset_dict["t"]["array"]) == 0:
            num_timesteps = len(dataset_dict["V"]["array"])
            dataset_dict["t"]["array"] = np.arange(start=0, stop=num_timesteps*dataset_dict["dt"], step=dataset_dict["dt"])

    return dataset_dict

def create_datasets_collection(training_datasets, testing_datasets):
    """
    Create a dictionary containing separate training and testing datasets.

    Args:
        training_datasets (dict): A dictionary of training datasets with dataset names as keys and dataset dictionaries as values.
        testing_datasets (dict): A dictionary of testing datasets with dataset names as keys and dataset dictionaries as values.

    Returns:
        dict: A dictionary containing two keys, "training" and "testing", each containing a sub-dictionary with keys "datasets" and "concatenated".
    """
    return {
        "training": {
            "datasets": training_datasets,
            "concatenated": {}
        },
        "testing": {
            "datasets": testing_datasets,
            "concatenated": {}
        }
    }

def concatenate_arrays(datasets_collection, category_list):
    """
    Concatenate the arrays for each dataset in a given category_list.

    Args:
        datasets_collection (dict): A dictionary containing datasets collections (e.g., "training" and "testing").
        category_list (list): A list of categories to concatenate (e.g., ["training", "testing"]).

    Returns:
        dict: A dictionary with concatenated arrays and individual datasets for each category specified in category_list.
    """
    return_dict = {}
    for category in category_list:
        return_dict[category] = {}
        return_dict[category]["individual_datasets"] = []
        new_t_list = []
        total_time_before_adding_dataset = 0
        total_time_after_adding_dataset = 0
        concatenated_V = np.zeros((0))
        concatenated_I_stim = np.zeros((0))
        for dataset_name, dataset in datasets_collection[category]["datasets"].items():
            total_time_before_adding_dataset = total_time_after_adding_dataset
            concatenated_V = np.concatenate((concatenated_V, dataset["V"]["array"]))
            concatenated_I_stim = np.concatenate((concatenated_I_stim, dataset["I_stim"]["array"]))
            total_time_after_adding_dataset += np.max(dataset["t"]["array"])
            new_t_list.append(total_time_before_adding_dataset+dataset["t"]["array"])
            dataset_copy = copy.deepcopy(dataset)
            dataset_copy["time_used"] = (total_time_before_adding_dataset, total_time_after_adding_dataset)
            return_dict[category]["individual_datasets"].append(copy.deepcopy(dataset_copy))
        new_t_array = np.concatenate(new_t_list)
        result = {"V": concatenated_V, "I_stim": concatenated_I_stim, "t": new_t_array}
        return_dict[category]["concatenated"] = result
        # plt.figure()
        # plt.plot(new_t_array)
        # plt.show()
    return return_dict


def convert_time_to_seconds(datasets_collection):
    """
    Convert all time arrays within the datasets_collection from milliseconds to seconds if they are in milliseconds,
    and leave them in seconds if they are already in seconds.

    Args:
        datasets_collection (dict): A dictionary containing training and testing datasets. Each dataset
                                    should have a 't' key containing a dictionary with an 'array' key for
                                    the time array and a 'units' key indicating whether the time array is
                                    in milliseconds ('ms') or seconds ('s').

    Returns:
        dict: The updated datasets_collection with time arrays converted to seconds if they were in milliseconds.
    """
    for dataset_type in ['training', 'testing']:
        for dataset_name, dataset in (datasets_collection[dataset_type])["datasets"].items():
            # Check if the time array is in milliseconds and convert to seconds if necessary
            if dataset['t']['units'] == 'ms':
                dataset['t']['array'] = dataset['t']['array'] / 1000.0
                dataset['t']['units'] = 's'
            # If it's already in seconds, do nothing
            elif dataset['t']['units'] == 's':
                continue
            else:
                raise ValueError(f"Unknown time units '{dataset['t']['units']}' in dataset '{dataset_name}'.")
    return datasets_collection

def create_collection_with_concatenated_dictionary_object(datasets_collection):
    """
    Concatenate data from the datasets collection and store it in the same dictionary.

    Args:
        datasets_collection (dict): A dictionary containing datasets collections (e.g., "training" and "testing").

    Returns:
        dict: The modified datasets_collection with concatenated data for each category ("training" and "testing").
    """
    # Convert all times to seconds (if already seconds, keep, but if in milliseconds, convert to seconds)
    datasets_collection = convert_time_to_seconds(datasets_collection)

    concatenated_data = concatenate_arrays(datasets_collection, ["training", "testing"])
    # Update datasets_collection with the concatenated data
    datasets_collection["training"]["concatenated"]["original_concatenation"] = concatenated_data["training"]["concatenated"]
    datasets_collection["testing"]["concatenated"]["original_concatenation"] = concatenated_data["testing"]["concatenated"]

    return datasets_collection


def create_time_delayed_embeddings_with_concatenated_dictionary_object(datasets_collection, param_dict):
    """
    Create time-delayed embeddings for voltage and stimulation current and store them in the datasets collection.

    Args:
        datasets_collection (dict): A dictionary containing datasets collections (e.g., "training" and "testing").
        param_dict (dict): A dictionary containing parameters for time-delayed embeddings.

    Returns:
        dict: The modified datasets_collection with time-delayed embeddings for each category ("training" and "testing").
    """
    T = param_dict["T"]
    D_E = param_dict["D_E"]

    for category in ["training", "testing"]:
        V_time_delayed = make_time_delay_embedding.time_delay_embedding(datasets_collection[category]["concatenated"]["original_concatenation"]["V"], T, D_E)
        I_stim_time_delayed = make_time_delay_embedding.time_delay_embedding(datasets_collection[category]["concatenated"]["original_concatenation"]["I_stim"], T, D_E)
        t_array_time_delay = make_time_delay_embedding.time_delay_embedded_time_series(datasets_collection[category]["concatenated"]["original_concatenation"]["t"], T, D_E)
        datasets_collection[category]["concatenated"]["time_delayed"] = {
            "V_time_delayed": V_time_delayed,
            "I_stim_time_delayed": I_stim_time_delayed,
            "T": T,
            "D_E": D_E,
            "t": (t_array_time_delay, T, D_E)
        }

    return datasets_collection


# def preprocess_datasets(datasets_collection, param_dict):
#     """
#     Preprocess the datasets by concatenating data and creating time-delayed embeddings for voltage and stimulation current.
#
#     Args:
#         datasets_collection (dict): A dictionary containing datasets collections (e.g., "training" and "testing").
#         param_dict (dict): A dictionary containing parameters for time-delayed embeddings.
#
#     Returns:
#         dict: The modified datasets_collection with concatenated data and time-delayed embeddings for each category ("training" and "testing").
#     """
#     # Step 1: Concatenate and store data
#     datasets_collection = create_collection_with_concatenated_dictionary_object(datasets_collection)
#
#     # Step 2: Create time-delayed embeddings and concatenate them
#     datasets_collection = create_time_delayed_embeddings_with_concatenated_dictionary_object(datasets_collection, param_dict)
#
#     return datasets_collection



# dataset_red171n2e1 = create_dataset_dict(
#     "Red171_Neuron2_Epoch1",
#     np.array([]),
#     np.array([]),
#     np.array([]),
#     0.00002,
#     "HVC_(RA)",
#     2016,
#     {"example1.txt": {"V": 0}, "example2.txt": {"I_stim": 0, "t": 1}},
#     "example.atf",
#     "mV",
#     "pA",
#     "s",
#     epoch=1,
#     extra_notes="ipsum lorum asdfasdfafd",
#     load_data=True,
# )

# # Example usage:
# dataset_CM_32425a75_2014_09_10_0013 = create_dataset_dict(
#     name="CM_2014_09_10_0013",
#     V=np.array([]),
#     I_stim=np.array([]),
#     t=np.array([]),
#     dt=0.000025,
#     neuron_type="CM 32425a75 (Phasic)",
#     collection_year=2014,
#     data_filepath_dict={"CM_data_and_prediction/Experimental/2014_09_10_0013_VIt.txt": {"V": 0, "I_stim": 1, "t": 2}},
#     data_filepath_original=None,
#     V_units="mV",
#     I_stim_units="pA",
#     t_units="s",
#     I_stim_name="Warped L63x(?)",
#     epoch=13,
#     extra_notes="",
#     load_data=True,
# )
#
# time_series_plot_and_save(
#     data=dataset_CM_32425a75_2014_09_10_0013["V"]["array"],
#     delta_t=dataset_CM_32425a75_2014_09_10_0013["dt"],
#     save_folder="temp_test_folder/",
#     title=dataset_CM_32425a75_2014_09_10_0013["name"] + " driven by " + dataset_CM_32425a75_2014_09_10_0013["I_stim_name"],
#     save_filename="test",
#     xlabel="t (" + dataset_CM_32425a75_2014_09_10_0013["t"]["units"] + ")",
#     ylabel="V (" + dataset_CM_32425a75_2014_09_10_0013["V"]["units"] + ")",
# )
#

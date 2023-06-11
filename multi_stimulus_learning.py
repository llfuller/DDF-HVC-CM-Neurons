import numpy as np
import copy
# Import make_time_delay_embedding
import sys
sys.path.append("./data_to_figure_code")
from data_loader import *
import make_general_plot

def estimate_hyperparameters(datasets_collection):
    # Insert logic here
    # Get time delay embedding of V(t) and I(t):
    T = 10  # Use estimate by AMI function
    D_E = 5  # Use estimate by FNN function
    beta = 10 ** -6  # Use estimate by k-fold cross validation
    R = 10 ** -3  # Use estimate by 1/(2*std_dev**2)
    hyperparams_estimated_dict = {"T":T,"D_E":D_E,"beta":beta,"R":R}
    return hyperparams_estimated_dict

def train_model(datasets_collection):
    """
    Add a docstring here
    """
    # Train using time delay embedded concatenated data:
    # ... same as before

def test_model(datasets_collection):
    """
    Add a docstring here
    """
    # Test using embedded concatenated data:
    # ... same as before

def main():
    T= 5
    D_E= 10
    data_directory = "data_to_figure_code/HVC_x_Red171_Neuron2_Epoch1_good_prediction/"
    run_hyperparam_search = False
    # Replace these lines with actual data loading
    dataset_L63x = create_dataset_dict("L63_x", np.array([]), np.array([]), np.array([]), 0.00002,
                                       "HVC_(X)", 2022, {data_directory+"epoch_1.txt": {"I_stim": 0, "V": 1}},
                                       {data_directory+"epoch_1.txt": {"I_stim": 0, "V": 1}},load_data=True)
    dataset_L63y = create_dataset_dict("L63_y", np.array([]), np.array([]), np.array([]), 0.00002,
                                       "HVC_(X)", 2022, {data_directory+"epoch_1.txt": {"I_stim": 0, "V": 1}},
                                       {data_directory+"epoch_1.txt": {"I_stim": 0, "V": 1}},load_data=True)
    dataset_L63y["V"]["array"] = 3*dataset_L63y["V"]["array"]
    dataset_Colpittsx = create_dataset_dict("Colpitts_x", np.array([]), np.array([]), np.array([]), 0.00002,
                                        "HVC_(X)", 2022, {data_directory+"epoch_1.txt": {"I_stim": 0, "V": 1}},
                                       {data_directory+"epoch_1.txt": {"I_stim": 0, "V": 1}},load_data=True)
    # Collect testing and training datasets into one dictionary object
    training_datasets = {dataset_L63x["name"]: dataset_L63x,
                         dataset_L63y["name"]: dataset_L63y
                         }
    testing_datasets = {dataset_Colpittsx["name"]: dataset_Colpittsx}
    datasets_collection = create_datasets_collection(training_datasets, testing_datasets)
    # Add concatendated data to dicitonary containing original training/testing data

    param_dict = {"T": T, "D_E": D_E}  # replace with hyperparams_estimated_dict entries when ready
    if run_hyperparam_search:
        # Estimate hyperparameters from datasets_collection
        hyperparams_estimated_dict = estimate_hyperparameters(datasets_collection)
        param_dict = {"T":hyperparams_estimated_dict["T"], "D_E":hyperparams_estimated_dict["D_E"]} # replace with hyperparams_estimated_dict entries when ready

    datasets_collection = preprocess_datasets(datasets_collection, param_dict)
    t_array = datasets_collection["training"]["concatenated"]["time_delayed"]["t"]

    train_model(datasets_collection) # using td-embedding
    test_model(datasets_collection) # using td-embedding

    print("Now making plots")

    make_general_plot.time_series_plot_and_save(
        data=datasets_collection["training"]["concatenated"]["time_delayed"]["V_time_delayed"][0],
        delta_t=0.02, # must be in ms
        save_folder="data_to_figure_code/temp_test_folder/",
        title="My test plot",
        save_filename="test2",
        xlabel="t (s)",
        ylabel="V (mV)",
    )

    print("Program is done!")

if __name__ == "__main__":
    main()

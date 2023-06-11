import numpy as np
import copy
# Import make_time_delay_embedding
import sys

import scipy.interpolate

sys.path.append("./data_to_figure_code")
from data_loader import *
import data_loader
import make_general_plot
import dynamical_systems_tools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import TimeDelay_Neuron_DDF_GaussianForm as TimeDelay_Neuron_DDF_GaussianForm_Added_V_Terms
import TimeDelay_Neuron_DDF_GaussianForm
import make_time_delay_embedding
from sklearn.cluster import KMeans
import os
import json
import make_FPS_compact

# import fnn
import save_utilities

folder_list = ["Train L63x,y,z and Colpittsx,y Test Colpittsz (time dilation 0p1)",
               "Train L63x,y,z and Colpittsx,y Test Colpittsz (time dilation 0p5)",
               "Train L63x,y,z and Colpittsx,y Test Colpittsz (time dilation 1)",
               "Train L63x,y,z and Colpittsx,y,z Test L63x Unseen Amplitude (time dilation 0p5)",
               "Train L63x Colpittsx, L63z, Colpittsz, Test Colpitts y Same Amplitude",
               "Train L63x Colpittsx, L63z, Colpittsz, Test Colpitts y Same Amplitude Modified RBF DDF (time dilation 0p5)",
               "Train L63x Colpittsx, L63z, Colpittsz, Test Colpitts y Same Amplitude Modified RBF DDF (time dilation 1p0)",
               "Train L63x Colpittsx, L63z, Colpittsz, Test L63x Same Amplitude (time dilation 1)",
               "Train L63x, Test L63x Same Amplitude (time dilation 1)",
               "Train L63x, Test L63x Next Time Segment (time dilation 3.5) Clark Paper Similarity"]
save_folder = f"data_to_figure_code/temp_test_folder/{folder_list[9]}/"#Train L63x L63y L63z Colpitts x Colpitts y Test Colpitts z (Time-Dilation 0pt2)/"
if not os.path.exists(save_folder + f"hyperparam_search/"):
    os.makedirs(save_folder + f"hyperparam_search/")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3D_time_delayed_projection(datasets_collection, category, T, D_E, save_folder):
    """
    Plots a 3D time-delayed projection of datasets with different colors and shapes for each dataset.

    Parameters:
    -----------
    datasets_collection: dict
        Collection of datasets with keys "training" and "testing" containing the datasets.
    category: str
        "training" or "testing", depending on which category of datasets to plot.
    T: int
        Time delay parameter for time-delay embedding.
    D_E: int
        Dimension of the time-delay embedding.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']  # Add more markers if needed
    color_idx = 0

    for dataset_name, dataset in (datasets_collection[category])["datasets"].items():
        V = dataset["V"]["array"]
        V_time_delayed = make_time_delay_embedding.time_delay_embedding(V, T, D_E)

        # Extract the coordinates for the three axes
        plotted_num_timesteps_delay = round(float(D_E) / 3)
        x = V_time_delayed[0, ::20]
        y = V_time_delayed[plotted_num_timesteps_delay, ::20]
        z = V_time_delayed[2 * plotted_num_timesteps_delay, ::20]

        # Create a 3D scatter plot with different colors and markers
        dataset_name_temp = dataset_name.replace("_","\_")
        ax.scatter(x, y, z, s=0.5, c=colors[color_idx], marker=markers[color_idx])#, label=r"$"+dataset_name_temp+"$")

        # Increment color index and marker index for the next dataset
        color_idx = (color_idx + 1) % len(colors)

    # Set axis labels
    ax.set_xlabel(r'$V(t)$')
    ax.set_ylabel(rf'V(t-{plotted_num_timesteps_delay}T$\Delta$t)')
    ax.set_zlabel(rf'V(t-{2 * plotted_num_timesteps_delay}T$\Delta$t)')

    # Add a legend
    # ax.legend()

    # Show the plot
    plt.show()
    plt.savefig(save_folder+"3D_composite_plot"+".png")
    plt.savefig(save_folder+"3D_composite_plot"+".pdf")


def find_drop_index(time_series, threshold):
    for index, value in enumerate(time_series):
        if value < threshold:
            return index
    return -1  # If the value doesn't drop below the threshold, return -1

def estimate_hyperparameters(datasets_collection):
    # Insert logic here
    # Get time delay embedding of V(t) and I(t):
    # T = 10  # Use estimate by AMI function
    # D_E = 5  # Use estimate by FNN function
    FNN_window = 1000
    directory_to_store_plots =     save_folder
    directory_to_store_txt_data = save_folder
    directory_to_store_FNN_data = save_folder

    print("hi")
    V_concat = datasets_collection["training"]["concatenated"]["original_concatenation"]["V"]
    I_concat = datasets_collection["training"]["concatenated"]["original_concatenation"]["I_stim"]
    t_concat = datasets_collection["training"]["concatenated"]["original_concatenation"]["t"]
    first_min_ami, max_lag, ami_scores = dynamical_systems_tools.estimate_tau(series=V_concat, max_lag=20, threshold_ratio=0.1)
    T = first_min_ami
    first_min_ami = 8
    fnn_ratios, min_dimension = dynamical_systems_tools.false_nearest_neighbors(series=V_concat, max_dimension=10, tau=first_min_ami, threshold=2.0)

    # max_dim, dim_dict = dynamical_systems_tools.false_nearest_neighbors(time_series=dataset[::1000], max_dim=6, tau=first_min_ami, r=10, metric='euclidean', chunk_size=1000)
    D_E =  min_dimension#find_drop_index(time_series=, threshold=0.06) # find first dimension in which FNN ratio drops below 0.06
    beta = 10 ** -2  # Use estimate by k-fold cross validation
    R = 5*10 ** -5  # Use estimate by 1/(2*std_dev**2)
    hyperparams_estimated_dict = {"T":T,"D_E":D_E,"beta":beta,"R":R}
    print(f"Estimated:{hyperparams_estimated_dict}")
    return hyperparams_estimated_dict

# def train_model(datasets_collection, centers, hyperparams_dict):
#     """
#     Add a docstring here
#     """
#     # Train using time delay embedded concatenated data:
#     # ... same as before
#     T   = hyperparams_dict["T"]
#     D_E = hyperparams_dict["D_E"]
#     R   = hyperparams_dict["R"]
#     beta = hyperparams_dict["beta"]
#     my_RBF_DDF_obj = TimeDelay_Neuron_DDF_GaussianForm.Gauss()
#     V_train = datasets_collection["training"]["concatenated"]["original_concatenation"]["V"]
#     I_stim_train = datasets_collection["training"]["concatenated"]["original_concatenation"]["I_stim"]
#     training_timesteps = len(V_train)-100
#
#     print("model training.")
#     difference_RHS_function = my_RBF_DDF_obj.FuncApproxF(V_train, training_timesteps, centers, beta, R, D_E, I_stim_train, T)
#     return [difference_RHS_function, my_RBF_DDF_obj]


def test_model(datasets_collection, difference_RHS_function, Prediction_length):
    """
    Add a docstring here
    """
    # Test using embedded concatenated data:
    # ... same as before
    I_stim_test = datasets_collection["testing"]["concatenated"]["original_concatenation"]["I_stim"]
    V_truth_test = datasets_collection["testing"]["concatenated"]["original_concatenation"]["V"]
    D_E = datasets_collection["testing"]["concatenated"]["time_delayed"]["D_E"]
    T = datasets_collection["testing"]["concatenated"]["time_delayed"]["T"]
    # V_pred = my_RBF_DDF_obj.PredictIntoTheFuture(flow_RHS_function, Prediction_length, I_stim_test[bias - 1:], V_truth_test[bias - 1 - (D_E - 1) * T:])

    n_start = T*D_E
    V_predicted = np.zeros(n_start+Prediction_length) #TODO: determine if I really need this plus 1
    # Set the first T*(D_E-1) values to be the initial condition
    V_predicted[0 : n_start+1] = V_truth_test[0 : n_start+1] # note that this is an array dim(V_pred)=(Prediction_length), not a 2-D array.
    assert(np.sum(V_predicted[n_start+1:])==0) # make sure rest of elements are zero so there are no data leaks
    # Predict Forward PreLength steps. The leading time delay dimension is predicted upon, and the previous steps of the leading term
    # become the values for the time delayed dimensions.
    for n in range(n_start, n_start+Prediction_length-2):
        V_predicted_n_tdelay = np.flip(V_predicted[n-n_start:n:T])
        # print(V_predicted_n_tdelay.shape)
        # print(I_stim_test[n:n+2].shape)
        V_predicted[n+1] = V_predicted[n] + difference_RHS_function(V_predicted_n_tdelay, I_stim_test[n:n+2])
    return V_predicted

def train_model(datasets_collection, centers, hyperparams_dict):
    """
    Add a docstring here
    """
    # Train using time delay embedded concatenated data:
    # ... same as before
    T   = hyperparams_dict["T"]
    D_E = hyperparams_dict["D_E"]
    R   = hyperparams_dict["R"]
    beta = hyperparams_dict["beta"]
    my_RBF_DDF_obj = TimeDelay_Neuron_DDF_GaussianForm.Gauss()
    V_train = datasets_collection["training"]["concatenated"]["original_concatenation"]["V"]
    I_stim_train = datasets_collection["training"]["concatenated"]["original_concatenation"]["I_stim"]
    training_timesteps = len(V_train)-100

    print("model training.")
    difference_RHS_function = my_RBF_DDF_obj.FuncApproxF(V_train, training_timesteps, centers, beta, R, D_E, I_stim_train, T)
    return [difference_RHS_function, my_RBF_DDF_obj]

def plot_and_save_training_and_testing(datasets_collection):
    # Requires a datasets_collection that has been through the "preprocess_datasets"
    make_general_plot.time_series_plot_and_save(
        data=datasets_collection["training"]["concatenated"]["time_delayed"]["V_time_delayed"][0],
        delta_t=0.02, # must be in ms
        save_folder=save_folder,
        title="Training V(t)",
        save_filename="train_V_NaKL",
        xlabel="t (s)",
        ylabel="V (mV)",
    )
    make_general_plot.time_series_plot_and_save(
        data=datasets_collection["training"]["concatenated"]["time_delayed"]["I_stim_time_delayed"][0],
        delta_t=0.02, # must be in ms
        save_folder=save_folder,
        title="Training I(t)",
        save_filename="train_I_NaKL",
        xlabel="t (s)",
        ylabel="I (pA)",
    )

    make_general_plot.time_series_plot_and_save(
        data=datasets_collection["testing"]["concatenated"]["time_delayed"]["V_time_delayed"][0],
        delta_t=0.02, # must be in ms
        save_folder=save_folder,
        title=r"Testing V$_{truth}$(t)",
        save_filename="test_V_truth_NaKL",
        xlabel="t (s)",
        ylabel="V (mV)",
    )
    make_general_plot.time_series_plot_and_save(
        data=datasets_collection["testing"]["concatenated"]["time_delayed"]["I_stim_time_delayed"][0],
        delta_t=0.02, # must be in ms
        save_folder=save_folder,
        title="Testing I(t)",
        save_filename="test_I_NaKL",
        xlabel="t (s)",
        ylabel="I (pA)",
    )


def main():

    T= 5
    D_E= 10
    data_directory = "Single_NaKL_Twin_Experiment_Original_data/"
    run_hyperparam_search = True

    print("Loading experimental data.")
    # Load in training data
    #"Train L63x Colpittsx, L63z, Colpittsz, Test Colpitts y Same Amplitude"
    names_to_select_training = [
        "NaKL_1_V(t)_Response_to_I_L63_x_time_dilation=3.5_(lower,upper)=(-55.0,48)pA"]
    loaded_json_training_dicts = data_loader.load_selected_json_dicts(file_path=data_directory+"all_dataset_dicts.json", names_to_select=names_to_select_training)
    loaded_json_training_dicts = data_loader.load_txt_array_into_dicts(loaded_json_training_dicts)

    # Load in testing data
    names_to_select_testing = ["NaKL_1_V(t)_Response_to_I_L63_x_time_dilation=3.5_(lower,upper)=(-55.0,48)pA_(t_start, t_stop)=(500, 1000)"]
    loaded_json_testing_dicts = data_loader.load_selected_json_dicts(file_path=data_directory+"all_dataset_dicts.json", names_to_select=names_to_select_testing)
    loaded_json_testing_dicts = data_loader.load_txt_array_into_dicts(loaded_json_testing_dicts)

    print("Preprocessing data.")
    # Collect testing and training datasets into two dictionary objects
    training_datasets = {}
    for a_dataset in loaded_json_training_dicts:
        training_datasets[a_dataset["name"]] = a_dataset
    testing_datasets = {}
    for a_dataset in loaded_json_testing_dicts:
        testing_datasets[a_dataset["name"]] = a_dataset

    # Add training and testing dictionaries to a collection
    datasets_collection = data_loader.create_datasets_collection(training_datasets, testing_datasets)

    # Add concatenated data to dictionary containing original training/testing data
    datasets_collection = data_loader.create_collection_with_concatenated_dictionary_object(datasets_collection)

    print("Estimating hyperparameters.")
    param_dict = {"T": T, "D_E": D_E}  # replace with hyperparams_estimated_dict entries when ready
    if run_hyperparam_search:
        # Estimate hyperparameters from datasets_collection
        hyperparams_estimated_dict = estimate_hyperparameters(datasets_collection)
        param_dict = {"T":hyperparams_estimated_dict["T"], "D_E":hyperparams_estimated_dict["D_E"]} # replace with hyperparams_estimated_dict entries when ready

    datasets_collection = data_loader.create_time_delayed_embeddings_with_concatenated_dictionary_object(datasets_collection, param_dict)

    # save FPS concatenated experimental data
    V_concatenated_interpolated = scipy.interpolate.interp1d(
        datasets_collection["training"]["concatenated"]["original_concatenation"]["t"],
        datasets_collection["training"]["concatenated"]["original_concatenation"]["V"])
    I_stim_concatenated_interpolated = scipy.interpolate.interp1d(
        datasets_collection["training"]["concatenated"]["original_concatenation"]["t"],
        datasets_collection["training"]["concatenated"]["original_concatenation"]["I_stim"])
    dt_FPS = 0.00002  # dt to use for interpolation in FPS
    # num_steps_interpolated = len(datasets_collection["training"]["concatenated"]["original_concatenation"]["t"])
    times_interpolated = np.arange(start=0,stop=datasets_collection["training"]["concatenated"]["original_concatenation"]["t"][-1],step=dt_FPS*1000)

    V_concatenated_interpolated_array = V_concatenated_interpolated(times_interpolated)
    I_stim_concatenated_interpolated_array =     I_stim_concatenated_interpolated(times_interpolated)
    # plt.figure()
    # plt.plot(datasets_collection["training"]["concatenated"]["original_concatenation"]["t"],datasets_collection["training"]["concatenated"]["original_concatenation"]["V"])
    # plt.plot(datasets_collection["training"]["concatenated"]["original_concatenation"]["t"],datasets_collection["training"]["concatenated"]["original_concatenation"]["I_stim"])
    # plt.show()
    # plt.figure()
    # plt.plot(times_interpolated,V_concatenated_interpolated_array)
    # plt.plot(times_interpolated,I_stim_concatenated_interpolated_array)
    # plt.show()
    make_FPS_compact.Fourier_Power_Spectrum_plot_and_save(V_concatenated_interpolated_array,
                                                          name = rf"FPS$_V$",
                                                          sampling_rate=1.0 / dt_FPS,
                                                          save_folder=save_folder + "FPS/",
                                                          title=rf"FPS$_V$",
                                                          save_filename=f"FPS_V_truth.png",
                                                          xlim=50)
    make_FPS_compact.Fourier_Power_Spectrum_plot_and_save(I_stim_concatenated_interpolated_array,
                                                          name = rf"FPS$_I$",
                                                          sampling_rate=1.0 / dt_FPS,
                                                          save_folder=save_folder + "FPS/",
                                                          title=rf"FPS$_I$",
                                                          save_filename=f"FPS_I_truth.png",
                                                          xlim=50)

    print("Finding centers.")
    num_centers = 500
    centers_path = save_folder+"centers.npy"
    if os.path.exists(centers_path):
        # Load centers from the file
        centers = np.load(centers_path)
    else:
        # Compute centers using KMeans and save them to the file
        centers = KMeans(n_clusters=num_centers, random_state=0).fit(datasets_collection["training"]["concatenated"]["time_delayed"]["V_time_delayed"].T).cluster_centers_
        np.save(centers_path, centers)

    # estimate ridge regression parameter
    # beta_est = estimate_Ridge_Regression_parameter(datasets_collection)

    # t_array = datasets_collection["training"]["concatenated"]["time_delayed"]["t"]
    #

    T = hyperparams_estimated_dict["T"]
    D_E = hyperparams_estimated_dict["D_E"]
    R_array = np.array([1e-3]) #np.logspace(-8, 4, num=13)  # 13 points from 10**-8 to 10**4
    beta_array = np.array([100]) #np.logspace(-8, 4, num=13)  # 13 points from 10**-8 to 10**4

    for R in R_array:
        for beta in beta_array:
            print("Starting loop")
            hyperparams_estimated_dict["R"]=R
            hyperparams_estimated_dict["beta"]=beta
            print("Estimating RHS difference map.")
            flow_RHS_function, my_RBF_DDF_obj = train_model(datasets_collection, centers, hyperparams_estimated_dict) # using td-embedding
            print("Predicting V(t).")
            Prediction_length = datasets_collection["testing"]["concatenated"]["original_concatenation"]["V"].shape[0] - (hyperparams_estimated_dict["T"]*hyperparams_estimated_dict["D_E"]) # subtract for buffer
            V_prediction = test_model(datasets_collection, flow_RHS_function, Prediction_length = Prediction_length) # using td-embedding
            print(V_prediction)
            print("location1")
            print("SDFKJSDFKJSDFLKJSDFL:KJSDFL:KJDSFLKJSDFLJK")
            print("Now making plots") #TODO: Fix the LaTeX Error here
            plot_3D_time_delayed_projection(datasets_collection,"training",param_dict["T"],param_dict["D_E"], save_folder)
            print("location2")
            plot_and_save_training_and_testing(datasets_collection)
            print("location3")
            print("Saving prediction plot")
            make_general_plot.time_series_plot_and_save(
                data=V_prediction,
                delta_t=0.02, # must be in ms
                save_folder=save_folder,
                title="Predicted V(t)",
                save_filename=f"predicted_V_NaKL_(T,D_E,R,beta)={(T,D_E,R,beta)}",
                xlabel="t (s)",
                ylabel="V (mV)"
            )
            plt.close("all")
            print("location4")


            np.savetxt(save_folder+f"predicted_V_NaKL_(T,D_E,R,beta)={(T,D_E,R,beta)}"+".txt", np.column_stack((datasets_collection["testing"]["concatenated"]["original_concatenation"]["t"],datasets_collection["testing"]["concatenated"]["original_concatenation"]["V"],V_prediction)))
            print("location5")


            plt.figure()
            plt.plot(datasets_collection["testing"]["concatenated"]["original_concatenation"]["t"]/1000,datasets_collection["testing"]["concatenated"]["original_concatenation"]["V"],label=r"V$_{truth}$", c="black")
            plt.plot(datasets_collection["testing"]["concatenated"]["original_concatenation"]["t"]/1000,V_prediction,label=r"V$_{pred}$", c="red", linestyle="--", linewidth=0.5)
            plt.ylabel("V (mV)")
            plt.xlabel("t (s)")
            plt.legend()
            plt.show()
            plt.savefig(save_folder+f"hyperparam_search/V_predicted_vs_truth_(T,D_E,R,beta)={(T,D_E,R,beta)}.png")
            plt.close("all")
            print("Program is done!")
            print("location6")

    data_used = {
        "names_to_select_training": names_to_select_training,
        "names_to_select_testing": names_to_select_testing,
    }
    filename = "datasets_selection_(train_and_test).json"
    file_path = os.path.join(save_folder, filename)
    with open(file_path, "w") as file:
        json.dump(data_used, file, indent=4)

if __name__ == "__main__":
    main()

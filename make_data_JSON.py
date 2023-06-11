import numpy as np
import sys
sys.path.append("./data_to_figure_code")
import data_loader


list_of_dictionary_datasets = []

############ MELIZA CM 2014

# Example usage:
dataset_CM_32425a75_2014_09_10_0013 = data_loader.create_dataset_dict(
    name="CM_32425a75_2014_09_10_0013",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.000025,
    neuron_type="CM 32425a75 (Phasic)",
    collection_year=2014,
    data_filepath_dict={"cm_ddf/data/32425a75/txt_V_I_t/2014_09_10_0013/epoch_13/epoch_13_segment_0.txt": {"V": 0, "I_stim": 1, "t": 2}}, # TODO: Determine whether this is segment 0 or not
    data_filepath_original={"cm_ddf/data/32425a75/2014_09_10_0013.abf": None},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="Complex Stimulus",
    epoch=13,
    extra_notes="Meliza calls this waveform the 'Complex' L63-derived waveform. It appears to be, from time-delay "
                "embedded 3-D projection of current in file, L63x or L63y which was distorted after generation."
                "This is one sweep from the epoch. For more information, see example-cells.ipynb that Meliza sent.",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_CM_32425a75_2014_09_10_0013)

dataset_CM_920061fe_2014_12_11_0010 = data_loader.create_dataset_dict(
    name="CM_920061fe_2014_12_11_0010",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.000025,
    neuron_type="CM 920061fe (Tonic)",
    collection_year=2014,
    data_filepath_dict={"cm_ddf/data/920061fe/txt_V_I_t/2014_12_11_0010/epoch_10/epoch_10_segment_0.txt": {"V": 0, "I_stim": 1, "t": 2}}, # TODO: Determine whether this is segment 0 or not
    data_filepath_original={"cm_ddf/data/920061fe/2014_12_11_0010.abf": None},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="Complex Stimulus",
    epoch=10,
    extra_notes="Meliza calls this waveform the 'Complex' L63-derived waveform. It appears to be, from time-delay "
                "embedded 3-D projection of current in file, L63x or L63y which was distorted after generation."
                "This is one sweep from the epoch. For more information, see example-cells.ipynb that Meliza sent.",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_CM_920061fe_2014_12_11_0010)


############ HVC 2016

dataset_HVC_RA_Neuron_52_Epoch1 = data_loader.create_dataset_dict(
    name="HVC_RA_Neuron_52_Epoch1",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_RA",
    collection_year=2022,
    data_filepath_dict={"HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 52/nidaq_Dev1_ai-0001_voltage.txt": {"V": 0},
                        "HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 52/nidaq_Dev1_ai-0001_current.txt": {"I_stim": 0}},
    data_filepath_original={"HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 52/nidaq_Dev1_ai-0001_voltage.txt": {"V": 0},
                            "HVC_ra_x_i_data_2016_2019/50 KhZ Recordings - 06_08_16/50KhZ-06_08_16/50KhZ-06_08_16/Neuron 52/nidaq_Dev1_ai-0001_current.txt": {"I_stim": 0}},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="?????", # TODO: Fill in
    epoch=1,
    extra_notes="Arij 2016 HVC Data."
                "Current is _________ with probable time-dilation ____. Epoch 1 is 5.3 seconds of data."
                "Neuron identities in HVC_ra_x_i_data_2016_2019/50%20KhZ%20Recordings%20-%2006_08_16/NeuronIdentities.pdf", # TODO: Fill in
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_RA_Neuron_52_Epoch1)


############ HVC 2019


############ HVC 2022
dataset_HVC_Lilac114_Neuron1_Epoch2 = data_loader.create_dataset_dict(
    name="Lilac114_Neuron1_Epoch2",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_X",
    collection_year=2022,
    data_filepath_dict={"Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/epoch_2.txt": {"I_stim": 0, "V": 1}},
    data_filepath_original=None,
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="L63x",
    epoch=2,
    extra_notes="Arij June 2022 Data. He said in an e-mail that everything from the June 2022 dataset should be HVC_X."
                "Current is L63x with probable time-dilation 0.5",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_Lilac114_Neuron1_Epoch2)


dataset_HVC_Lilac114_Neuron1_Epoch3 = data_loader.create_dataset_dict(
    name="Lilac114_Neuron1_Epoch3",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_X",
    collection_year=2022,
    data_filepath_dict={"Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/epoch_3.txt": {"I_stim": 0, "V": 1}},
    data_filepath_original={"Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/epoch_3.txt": {"I_stim": 0, "V": 1}},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="L63x",
    epoch=3,
    extra_notes="Arij June 2022 Data. He said in an e-mail that everything from the June 2022 dataset should be HVC_X"
                "Current is L63x with probable time-dilation 0.5",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_Lilac114_Neuron1_Epoch3)


dataset_HVC_Lilac114_Neuron1_Epoch5 = data_loader.create_dataset_dict(
    name="Lilac114_Neuron1_Epoch5",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_X",
    collection_year=2022,
    data_filepath_dict={"Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/epoch_5.txt": {"I_stim": 0, "V": 1}},
    data_filepath_original={"Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 1/epoch_5.txt": {"I_stim": 0, "V": 1}},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="L63x",
    epoch=3,
    extra_notes="Arij June 2022 Data. He said in an e-mail that everything from the June 2022 dataset should be HVC_X"
                "Current is L63x with probable time-dilation 0.5",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_Lilac114_Neuron1_Epoch5)


dataset_HVC_Lilac114_Neuron2_Epoch2 = data_loader.create_dataset_dict(
    name="Lilac114_Neuron2_Epoch2",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_X",
    collection_year=2022,
    data_filepath_dict={"Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 2/epoch_2.txt": {"I_stim": 0, "V": 1}},
    data_filepath_original={"Data2022-50KhZ/7-7-2022/Lilac 114/Neuron 2/epoch_2.txt": {"I_stim": 0, "V": 1}},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="L63x",
    epoch=2,
    extra_notes="Arij June 2022 Data. He said in an e-mail that everything from the June 2022 dataset should be HVC_X"
                "Current is L63x with probable time-dilation 0.2",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_Lilac114_Neuron2_Epoch2)


dataset_HVC_Lilac242_Neuron1_Epoch1 = data_loader.create_dataset_dict(
    name="Lilac242_Neuron1_Epoch1",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_X",
    collection_year=2022,
    data_filepath_dict={"Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/epoch_1.txt": {"I_stim": 0, "V": 1}},
    data_filepath_original={"Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/epoch_1.txt": {"I_stim": 0, "V": 1}},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="L63x",
    epoch=1,
    extra_notes="Arij June 2022 Data. He said in an e-mail that everything from the June 2022 dataset should be HVC_X"
                "Current is L63x with probable time-dilation 0.2",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_Lilac242_Neuron1_Epoch1)


dataset_HVC_Lilac242_Neuron1_Epoch9 = data_loader.create_dataset_dict(
    name="Lilac242_Neuron1_Epoch9",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_X",
    collection_year=2022,
    data_filepath_dict={"Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/epoch_9.txt": {"I_stim": 0, "V": 1}},
    data_filepath_original={"Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/epoch_9.txt": {"I_stim": 0, "V": 1}},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="L63x",
    epoch=9,
    extra_notes="Arij June 2022 Data. He said in an e-mail that everything from the June 2022 dataset should be HVC_X"
                "Current is L63x with probable time-dilation 0.2",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_Lilac242_Neuron1_Epoch9)


dataset_HVC_Lilac242_Neuron1_Epoch10 = data_loader.create_dataset_dict(
    name="Lilac242_Neuron1_Epoch10",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_X",
    collection_year=2022,
    data_filepath_dict={"Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/epoch_10.txt": {"I_stim": 0, "V": 1}},
    data_filepath_original={"Data2022-50KhZ/7-7-2022/Lilac 242/Neuron 1/epoch_10.txt": {"I_stim": 0, "V": 1}},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="L63x",
    epoch=10,
    extra_notes="Arij June 2022 Data. He said in an e-mail that everything from the June 2022 dataset should be HVC_X"
                "Current is L63x with probable time-dilation 0.2",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_Lilac242_Neuron1_Epoch10)


dataset_HVC_Red171_Neuron2_Epoch1 = data_loader.create_dataset_dict(
    name="Red171_Neuron2_Epoch1",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_X",
    collection_year=2022,
    data_filepath_dict={"Data2022-50KhZ/7-7-2022/Red 171/Neuron 2/epoch_1.txt": {"I_stim": 0, "V": 1}},
    data_filepath_original={"Data2022-50KhZ/7-7-2022/Red 171/Neuron 2/epoch_1.txt": {"I_stim": 0, "V": 1}},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="Colpitts_x",
    epoch=1,
    extra_notes="Arij June 2022 Data. He said in an e-mail that everything from the June 2022 dataset should be HVC_X"
                "Current is Colpitts_x with probable time-dilation 0.5",
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_Red171_Neuron2_Epoch1)


dataset_HVC_Lilac157_Neuron1_Epoch1 = data_loader.create_dataset_dict(
    name="Lilac157_Neuron1_Epoch1",
    V=np.array([]),
    I_stim=np.array([]),
    t=np.array([]),
    dt=0.00002,
    neuron_type="HVC_RA(?)",
    collection_year=2022,
    data_filepath_dict={"Data2022-50KhZ/11-30-2022/Lilac 157/Neuron 1/epoch_1.txt": {"I_stim": 0, "V": 1}},
    data_filepath_original={"Data2022-50KhZ/11-30-2022/Lilac 157/Neuron 1/epoch_1.txt": {"I_stim": 0, "V": 1}},
    V_units="mV",
    I_stim_units="pA",
    t_units="s",
    I_stim_name="?????",
    epoch=1,
    extra_notes="Arij Nov 2022 Data. He said in an e-mail that this is probably HVC_RA, but it's not definitive."
                "Current is _________ with probable time-dilation ____", # TODO: Fill in
    load_data=True,
)
list_of_dictionary_datasets.append(dataset_HVC_Lilac157_Neuron1_Epoch1)


############ HVC 2023

data_loader.update_json_file("neuron_data_reference.json", list_of_dictionary_datasets)
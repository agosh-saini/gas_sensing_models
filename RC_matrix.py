##################
# Author: Agosh Saini
# Date:2024-10-25
# Contact: contact@agoshsaini.com
##################
# Description: This script is used to generate the RC matrix for the given data files
##################

#### IMPORTS ####

import numpy as np
import json
import os
import ast

#### CLASSES ####

class matrix_generator:

    def __init__(self, data_folder, gas_index):

        '''
        This function is used to initialize the matrix generator object
        '''

        self.data_folder = data_folder
        self.data_files = os.listdir(data_folder)
        self.gas_index = gas_index

    def get_sensor_list(self):

        '''
        This function is used to get the list of sensors in the data files

        Returns:
            sensor_list: The list of sensors in the data files
        '''

        sensor_list = []

        for data_file in self.data_files:

            with open(os.path.join(self.data_folder, data_file), 'r') as file:

                data = json.load(file)
                sensor = data["Sensor Type"]
                
                if sensor not in sensor_list:
                    sensor_list.append(sensor)
        
        self.sensor_list = sensor_list

        return sensor_list

    def extract_data(self, sensor_key):

        '''
        This function is used to extract the data from the data files

        Parameters:
            sensor_key: The key of the sensor data in the data file's JSON

        Returns:
            signal_arr: The signal matrix of the data
            gas_ppm_arr: The gas ppm matrix of the data
            gas_type_arr: The gas type matrix of the data
        '''

        signal_list = []
        gas_ppm_list = []
        gas_type_list = []

        for data_file in self.data_files:

            with open(os.path.join(self.data_folder, data_file), 'r') as file:
                data = json.load(file)

                if data["Sensor Type"] == sensor_key:

                    signal = [data["RC_on"]["Delta_R"], data["RC_off"]["Delta_R"], data["RC_on"]["tau"], data["RC_off"]["tau"]]
                    gas_type = data["Analyte"]
                    gas_ppm = data["ppm"]

                    signal_list.append(signal)
                    gas_ppm_list.append(gas_ppm)
                    gas_type_list.append(gas_type)

        signal_arr = np.array(signal_list)
        gas_ppm_arr = np.array(gas_ppm_list)
        gas_type_arr = np.array(gas_type_list)

        return signal_arr, gas_ppm_arr, gas_type_arr
    
    def arrange_ppms(self, gas_ppm_arr, gas_type_arr, gas_types=None):

        '''
        This function is used to arrange the gas ppm matrix and gas type matrix

        Parameters:
            gas_ppm_arr: The gas ppm array of the data
            gas_type_arr: The gas type array of the data

        Returns:
            arranged_gas_ppm_arr: The arranged gas ppm matrix of the data
            gas_types: The list of gas types
        '''

        if gas_types is None:
            gas_types = np.unique(gas_type_arr)

        arranged_gas_ppm_arr = np.zeros((len(gas_ppm_arr), len(gas_types)))
        
        # Loop over each unique gas type
        for idx, gas_type in enumerate(gas_types):
            
            # Find indices in gas_type_arr where the current gas type appears
            indices = [i for i, g in enumerate(gas_type_arr) if g == gas_type]

            # Place corresponding ppm values in the column for this gas type
            for i in indices:
                arranged_gas_ppm_arr[i, idx] = gas_ppm_arr[i]

        return arranged_gas_ppm_arr, gas_types

    def process_replicates(self, sensor_key):

        '''
        TO BE CREATED: This function is used to process the replicates of the data files

        Parameters:
            sensor_key: The key of the sensor data in the data file's JSON

        Returns:
            signal: The signal matrix of the data
            gas_ppm: The gas ppm matrix of the data
        '''

        # Implementation pending

        pass

    def generate_RC_matrix(self, signal, gas_ppm):

        '''
        This function is used to generate the RC matrix for the given data files

        Parameters:
            signal: The signal matrix of the data
            gas_ppm: The gas ppm matrix of the data

        Returns:
            RC_matrix: The RC matrix for the given data files
        '''
        
        RC_matrix, _, _, _ = np.linalg.lstsq(signal, gas_ppm, rcond=None)

        return RC_matrix

class meta_regressor:

    def __init__(self, processed_data_folder, RC_matrix_folder, 
                 gas_types_folder, arranged_gas_ppm_folder):
        
        '''
        This function is used to initialize the meta regressor object
        '''

        self.processed_data_folder = processed_data_folder
        self.RC_matrix_folder = RC_matrix_folder
        self.gas_types_folder = gas_types_folder
        self.arranged_gas_ppm_folder = arranged_gas_ppm_folder

    def load_RC_matrices(self):

        '''
        This function is used to load the RC matrices from the folder

        Returns:
            RC_matrices: The RC matrices of the data files
        '''

        RC_matrices = []

        for RC_matrix_file in os.listdir(self.RC_matrix_folder):

            with open(os.path.join(self.RC_matrix_folder, RC_matrix_file), "rb") as file:
                RC_matrix = np.load(file)
                RC_matrices.append(RC_matrix)
        
        RC_matrices = np.stack(RC_matrices, axis=2)

        return RC_matrices

    def load_arranged_gas_ppm(self):
        
        '''
        This function is used to load the arranged gas ppm from the folder

        Returns:
            arranged_gas_ppm: The arranged gas ppm of the data files
        '''

        arranged_gas_ppm_list = []

        for arranged_gas_ppm_file in os.listdir(self.arranged_gas_ppm_folder):

            with open(os.path.join(self.arranged_gas_ppm_folder, arranged_gas_ppm_file), "rb") as file:
                arranged_gas_ppm = np.load(file)
                arranged_gas_ppm_list.append(arranged_gas_ppm)
        
        arranged_gas_ppm = np.concatenate(arranged_gas_ppm_list, axis=0)

        return arranged_gas_ppm

    def load_gas_types(self):

        '''
        This function is used to load the gas types from the folder

        Returns:
            gas_types: The gas types of the data files
        '''

        gas_types_list = []

        for gas_types_file in os.listdir(self.gas_types_folder):

            with open(os.path.join(self.gas_types_folder, gas_types_file), "rb") as file:
                gas_types = np.load(file)
                gas_types_list.append(gas_types)
        
        gas_types = np.concatenate(gas_types_list)

        return gas_types
    
    def predict(self, RC_matrices, signal):

        '''
        This function is used to predict the gas ppm for the given signal

        Parameters:
            RC_matrices: The RC matrices of the data files, shape (n_features, n_gases, n_matrices)
            signal: The signal array of the data, shape (n_features,)

        Returns:
            gas_ppm: The gas ppm vector of the data
        '''

        # Ensure signal is a 1D array
        signal = np.array(signal).flatten()  # Shape (n_features,)

        n_matrices = RC_matrices.shape[2]
        n_gases = RC_matrices.shape[1]

        gas_ppm_results = np.zeros((n_gases, n_matrices))

        for i in range(n_matrices):

            RC_matrix = RC_matrices[:, :, i]

            # Predict the gas ppm
            gas_ppm = np.dot(signal, RC_matrix)  # signal shape (n_features,), RC_matrix shape (n_features, n_gases)
            gas_ppm_results[:, i] = gas_ppm

        gas_ppm_mean = np.mean(gas_ppm_results, axis=1)  # Mean over matrices

        return gas_ppm_mean

#### MAIN ####

if __name__ == "__main__":

    # Define the data folder
    data_folder = "json_folder"
    gas_index = ['water', 'EtOH']

    # Create the matrix generator object
    matrix_gen = matrix_generator(data_folder, gas_index)
    sensor_list = matrix_gen.get_sensor_list()

    # Define the processed data folders
    processed_data_folder = "processed_data"
    RC_matrix_folder = os.path.join(processed_data_folder, "RC_matrices")
    arranged_gas_ppm_folder = os.path.join(processed_data_folder, "arranged_gas_ppm")
    gas_types_folder = os.path.join(processed_data_folder, "gas_types")
    signal_folder = os.path.join(processed_data_folder, "signal")

    # Create the folders if they do not exist
    os.makedirs(RC_matrix_folder, exist_ok=True)
    os.makedirs(arranged_gas_ppm_folder, exist_ok=True)
    os.makedirs(gas_types_folder, exist_ok=True)
    os.makedirs(signal_folder, exist_ok=True)

    # Generate the RC matrices for the sensors
    for sensor in sensor_list:
        # Extract the data from the data files
        signal, gas_ppm, gas_type_arr = matrix_gen.extract_data(sensor)

        # Arrange the gas ppm matrix and gas type matrix
        arranged_gas_ppm, gas_types = matrix_gen.arrange_ppms(gas_ppm, gas_type_arr)

        # Generate the RC matrix
        RC_matrix = matrix_gen.generate_RC_matrix(signal, arranged_gas_ppm)

        # Save the RC matrix and associated data
        np.save(os.path.join(RC_matrix_folder, f"{sensor}.npy"), RC_matrix)
        np.save(os.path.join(arranged_gas_ppm_folder, f"{sensor}.npy"), arranged_gas_ppm)
        np.save(os.path.join(gas_types_folder, f"{sensor}.npy"), gas_types)
        np.save(os.path.join(signal_folder, f"{sensor}.npy"), signal)
    
    # Create the meta_regressor object
    meta_reg = meta_regressor(processed_data_folder, RC_matrix_folder, gas_types_folder, arranged_gas_ppm_folder)

    # Load the data
    RC_matrices = meta_reg.load_RC_matrices()
    gas_types = meta_reg.load_gas_types()
    arranged_gas_ppm = meta_reg.load_arranged_gas_ppm()

    # Parse the input signal
    signal_str = input("Enter the signal as array [on_R, off_R, on_tau, off_tau]: ")
    signal = np.array(ast.literal_eval(signal_str), dtype=float)

    # Predict gas ppm
    gas_ppm = meta_reg.predict(RC_matrices, signal)
    print("Predicted gas ppm values:", gas_ppm)
    print("Gas Index:", gas_types)

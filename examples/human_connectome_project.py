import os
import pandas as pd
import numpy as np

# Define the folder containing the .txt files
folder_path = ('/spm-data/vault-data3/mmll/data/HCP/100_nodes_times_series/3T_HCP1200_MSMAll_d100_ts2')

# Initialize a list to store correlation matrices
correlation_matrices = []

# Initialize a list to store subject IDs
subject_ids = []

file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Sort the filenames by subject ID (assuming subject IDs are in the filename before .txt)
file_list.sort()  # This will sort alphabetically, which works if subject IDs are formatted consistently

# Loop through all files in the specified folder
for filename in file_list:
    print(filename)
    # Construct the full file path
    file_path = os.path.join(folder_path, filename)

    # Read the text file into a DataFrame
    df = pd.read_csv(file_path, sep='\s+', header=None)

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Append the correlation matrix to the list
    correlation_matrices.append(correlation_matrix.values)

    # Extract subject ID from the filename (remove the .txt extension)
    subject_id = filename[:-4]  # Exclude the last 4 characters ('.txt')
    subject_ids.append(subject_id)

# Convert the list of correlation matrices into a 3D numpy array
correlation_array = np.array(correlation_matrices)

# Create a DataFrame for subject IDs
subject_ids_df = pd.DataFrame(subject_ids, columns=['Subject_ID'])

# Save the numpy array and DataFrame to disk
np.save('/spm-data/vault-data3/mmll/data/HCP/100_nodes_times_series/functional_connectivity.npy', correlation_array)  # Save as .npy file
subject_ids_df.to_csv('/spm-data/vault-data3/mmll/data/HCP/100_nodes_times_series/subject_ids.csv', index=False)  # Save as .csv file


print()
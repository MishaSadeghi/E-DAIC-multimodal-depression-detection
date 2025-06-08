import numpy as np
import pandas as pd
import re
import pyarrow as pa
import pyarrow.feather as feather

# # Reading the extracted features from the audio file using the saved CSV file 
# csv_file_path = 'extracted_features_from_audio_trimmed.csv'
# df_audio_features = pd.read_csv(csv_file_path)

# # List of columns to be corrected
# columns_to_correct = ['raw_f0_values', 'raw_formant_values', 'mfccs', 'mfccs_mean']

# # Iterate over each column
# for column_name in columns_to_correct:
#     # Get the column from the DataFrame
#     column_values = df_audio_features[column_name]

#     # Iterate over each element in the column
#     for i, element in enumerate(column_values):
#         if isinstance(element, str):
#             # Remove unwanted characters
#             cleaned_instance = re.sub(r'[\[\]]', '', element)

#             # Replace 'nan' with 'NaN'
#             cleaned_instance = cleaned_instance.replace('nan', 'NaN')

#             # Split the string into individual values
#             values = cleaned_instance.split(',')

#             # Convert each number from string to float
#             float_values = [float(value) if value != 'NaN' else np.nan for value in values]

#             # Calculate the mean of non-NaN values
#             mean_value = np.nanmean(float_values)

#             # Replace NaN values with the mean
#             float_values = [mean_value if np.isnan(value) else value for value in float_values]

#             # Convert the list of float values to a NumPy array
#             float_values_array = np.array(float_values)

#             # Assign the array back to the DataFrame
#             df_audio_features.at[i, column_name] = float_values_array

#             # Print the type and shape
#             print(f"Element {i} in column {column_name}: Type - {type(df_audio_features.at[i, column_name])}, Shape - {df_audio_features.at[i, column_name].shape}")
#         else:
#             # Print the type and shape for non-string elements
#             print(f"Element {i} in column {column_name}: Type - {type(element)}")

# # Apply np.array conversion to the specified columns
# df_audio_features['mfccs_mean'] = df_audio_features['mfccs_mean'].apply(np.array)
# df_audio_features['raw_f0_values'] = df_audio_features['raw_f0_values'].apply(np.array)
# df_audio_features['raw_formant_values'] = df_audio_features['raw_formant_values'].apply(np.array)
# df_audio_features['mfccs'] = df_audio_features['mfccs'].apply(np.array)

# # Save the DataFrame to a Feather file
# output_file_path = 'corrected_extracted_features_from_audio_trimmed.feather'
# df_audio_features.to_feather(output_file_path)

# Load the DataFrame from the saved file
file_path = 'corrected_extracted_features_from_audio_trimmed.feather'
df_audio_features = feather.read_feather(file_path)

# Check types and shapes of all columns
for column_name in df_audio_features.columns:
    column_values = df_audio_features[column_name]

    if isinstance(column_values[0], np.ndarray):
        print(f"Column '{column_name}': Type - {type(column_values[0])}, Shape - {column_values[0].shape}")
    else:
        print(f"Column '{column_name}': Type - {type(column_values[0])}")


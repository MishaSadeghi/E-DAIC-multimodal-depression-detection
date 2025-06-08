import pandas as pd
import numpy as np
import json
import ast
import re
from io import StringIO
import pyarrow.feather as feather
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew

# Load the DataFrame from the saved file
file_path = 'corrected_extracted_features_from_audio_trimmed.feather'
df_audio_features = feather.read_feather(file_path)

def calculate_statistics(array):
    if np.isnan(array).any():
        # Handle arrays with NaN values
        array = array[~np.isnan(array)]
    
    mean_value = np.mean(array)
    std_dev = np.std(array)
    skewness = skew(array)

    return mean_value, std_dev, skewness

# Apply the function to your DataFrame columns
df_audio_features[['mfccs_mean_mean', 'mfccs_mean_std', 'mfccs_mean_skew']] = df_audio_features['mfccs_mean'].apply(calculate_statistics).apply(pd.Series)
df_audio_features[['raw_f0_values_mean', 'raw_f0_values_std', 'raw_f0_values_skew']] = df_audio_features['raw_f0_values'].apply(calculate_statistics).apply(pd.Series)
df_audio_features[['raw_formant_values_mean', 'raw_formant_values_std', 'raw_formant_values_skew']] = df_audio_features['raw_formant_values'].apply(calculate_statistics).apply(pd.Series)
df_audio_features[['mfccs_mean', 'mfccs_std', 'mfccs_skew']] = df_audio_features['mfccs'].apply(calculate_statistics).apply(pd.Series)

# Drop the original array columns
df_audio_features = df_audio_features.drop(['mfccs_mean', 'raw_f0_values', 'raw_formant_values', 'mfccs'], axis=1)

# Check types and shapes of all columns
for column_name in df_audio_features.columns:
    column_values = df_audio_features[column_name]

    if isinstance(column_values[0], np.ndarray):
        print(f"Column '{column_name}': Type - {type(column_values[0])}, Shape - {column_values[0].shape}")
    else:
        print(f"Column '{column_name}': Type - {type(column_values[0])}")
       
print('df_audio_features: ', df_audio_features.head())

# column_types = df_audio_features.dtypes
# column_types_dict = column_types.to_dict()
# for column, data_type in column_types_dict.items():
#     print(f'{column}: {data_type}')

# Reading the labels files
labels_dev_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/dev_split.csv')
labels_dev_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_train_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/train_split.csv')
labels_train_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_test_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/test_split.csv')
labels_test_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

# Combine all labels into one DataFrame
labels_df = pd.concat([labels_dev_df, labels_train_df, labels_test_df], ignore_index=True)
print('All Labels: ', labels_df.head())

# Merge features with labels for train, dev, and test sets
merged_train = pd.merge(df_audio_features, labels_train_df, on='id')
merged_dev = pd.merge(df_audio_features, labels_dev_df, on='id')
merged_test = pd.merge(df_audio_features, labels_test_df, on='id')

# Extract X (features) and y (target) for train, dev, and test sets
X_train = merged_train.drop(['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity'], axis=1)
y_train = merged_train['PHQ_Score']
X_dev = merged_dev.drop(['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity'], axis=1)
y_dev = merged_dev['PHQ_Score']
X_test = merged_test.drop(['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity'], axis=1)
y_test = merged_test['PHQ_Score']

# Replace 'Gender' column with integers 0 and 1
# df['Gender'] = df['Gender'].replace({'female': 0, 'male': 1})
# df['Gender'] = df['Gender'].replace({'female ': 0, 'male ': 1})

print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_dev: {y_dev.shape}')
print(f'Shape of y_test: {y_test.shape}')
print('X_train: ', X_train.head())

# Impute missing values if needed
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_dev_imputed = imputer.transform(X_dev)
X_test_imputed = imputer.transform(X_test)

print(f'Shape of X_train_imputed: {X_train_imputed.shape}')
print(f'Shape of X_dev_imputed: {X_dev_imputed.shape}')
print(f'Shape of X_test_imputed: {X_test_imputed.shape}')

# # Define the number of features to select
# num_features_to_select = 20  # Adjust as needed

# # Initialize SelectKBest with f_regression score function
# feature_selector = SelectKBest(score_func=f_regression, k=num_features_to_select)

# # Fit and transform on the imputed training set
# X_train_selected = feature_selector.fit_transform(X_train_imputed, y_train)

# # Transform the dev and test sets
# X_dev_selected = feature_selector.transform(X_dev_imputed)
# X_test_selected = feature_selector.transform(X_test_imputed)

# print(f'Shape of X_train_selected: {X_train_selected.shape}')
# print(f'Shape of X_dev_selected: {X_dev_selected.shape}')
# print(f'Shape of X_test_selected: {X_test_selected.shape}')

# Initialize RandomForestRegressor
regressor = RandomForestRegressor(random_state=42)

# Train the model on the selected features of the training set
regressor.fit(X_train_imputed, y_train)

# Predict on the development set
y_dev_pred = regressor.predict(X_dev_imputed)
# Evaluate the model on the development set
mae_dev = mean_absolute_error(y_dev, y_dev_pred)
print(f'MAE on dev set: {mae_dev}')
rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))
print(f'RMSE on dev set: {rmse_dev}')

# Predict on the test set
y_test_pred = regressor.predict(X_test_imputed)
# Evaluate the model on the test set
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE on test set: {mae_test}')
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f'RMSE on test set: {rmse_test}')


# ------------------------------------------------------------------------------------------------
# # Binary Classification based on PHQ_Binary

# # Merge features with labels for train, dev, and test sets
# merged_train = pd.merge(df_audio_features, labels_train_df, on='id')
# merged_dev = pd.merge(df_audio_features, labels_dev_df, on='id')
# merged_test = pd.merge(df_audio_features, labels_test_df, on='id')

# # Extract X (features) and y (target) for train, dev, and test sets
# X_train = merged_train.drop(['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity'], axis=1)
# y_train = merged_train['PHQ_Binary']
# X_dev = merged_dev.drop(['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity'], axis=1)
# y_dev = merged_dev['PHQ_Binary']
# X_test = merged_test.drop(['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity'], axis=1)
# y_test = merged_test['PHQ_Binary']

# # Train a Random Forest classifier
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# # Get feature importances
# feature_importances = clf.feature_importances_

# # Get feature names
# feature_names = X_train.columns

# # Print feature names and their importances
# for feature_name, importance in zip(feature_names, feature_importances):
#     print(f"{feature_name}: {importance:.4f}")

# # ------------------------------------------------------------------------------------------------
# # Classification Reports

# # Make predictions on the dev set
# y_dev_pred = clf.predict(X_dev)
# print("Classification Report on Dev Set:")
# print(classification_report(y_dev, y_dev_pred))

# # Make predictions on the test set
# y_test_pred = clf.predict(X_test)
# print("Classification Report on Test Set:")
# print(classification_report(y_test, y_test_pred))

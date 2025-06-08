import os
import gc
import sys
import numpy as np
import scipy as sp
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tsfresh import select_features
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_predict

# Reading the labels files
labels_dev_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/dev_split.csv')
labels_dev_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_train_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/train_split.csv')
labels_train_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_test_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/test_split.csv')
labels_test_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

# Combine all labels into one DataFrame
labels_df = pd.concat([labels_dev_df, labels_train_df, labels_test_df], ignore_index=True)
print(labels_df)

# Reading Open Face data from the CSV files
directory_path_openface = '/home/hpc/empk/empk004h/depression-detection/data/DAIC_openface_features/'
dfs = []

# Iterate through the CSV files in the directory
for filename in os.listdir(directory_path_openface):
    if filename.endswith('.csv'):
        participant_id = filename.split('_')[0]  # Extract participant ID from filename
        file_path = os.path.join(directory_path_openface, filename)        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)        
        # Insert the 'id' column with the participant ID
        df.insert(0, 'id', int(participant_id))  # Ensure 'id' is of int type       
        dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
df_openface = pd.concat(dfs, ignore_index=True)

# Filtering based on confidence
indices_to_remove = df_openface[df_openface['confidence'] < 0.9].index
df_openface = df_openface.drop(index=indices_to_remove)
df_openface = df_openface.sort_values(by=['id', 'frame'])
df_openface = df_openface.reset_index(drop=True)
df_openface = df_openface.drop(['timestamp', 'confidence', 'success'], axis=1)
print('df_openface : ', df_openface)

# Extract participant IDs for train, dev, and test sets from label files
train_ids = labels_train_df['id'].unique()
dev_ids = labels_dev_df['id'].unique()
test_ids = labels_test_df['id'].unique()

df_openface_train = df_openface[df_openface['id'].isin(train_ids)].copy()
df_openface_dev = df_openface[df_openface['id'].isin(dev_ids)].copy()
df_openface_test = df_openface[df_openface['id'].isin(test_ids)].copy()

print('df_openface_train : ', df_openface_train)
print('df_openface_dev : ', df_openface_dev)
print('df_openface_test : ', df_openface_test)

minimal_features_train = extract_features(df_openface_train, column_id='id', column_sort='frame', default_fc_parameters=MinimalFCParameters())
minimal_features_dev = extract_features(df_openface_dev, column_id='id', column_sort='frame', default_fc_parameters=MinimalFCParameters())
minimal_features_test = extract_features(df_openface_test, column_id='id', column_sort='frame', default_fc_parameters=MinimalFCParameters())

print("Columns in minimal_features_dev:", minimal_features_dev.columns)

# # Drop 'id' and 'frame' columns after feature extraction
# minimal_features_train = minimal_features_train.drop(['id', 'frame'], axis=1)
# minimal_features_dev = minimal_features_dev.drop(['id', 'frame'], axis=1)
# minimal_features_test = minimal_features_test.drop(['id', 'frame'], axis=1)

# Impute the result (fill NaN values)
minimal_features_train = impute(minimal_features_train)
minimal_features_dev = impute(minimal_features_dev)
minimal_features_test = impute(minimal_features_test)

print("Columns in minimal_features_dev after impute:", minimal_features_dev.columns)

# -----------------------------------------------------------------------------
#  PHQ Binary Classification

# X_train = minimal_features_train
# y_train = labels_train_df['PHQ_Binary']

# X_dev = minimal_features_dev
# y_dev = labels_dev_df['PHQ_Binary']

# X_test = minimal_features_test
# y_test = labels_test_df['PHQ_Binary']

# # Perform feature selection and classification on the train set
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Evaluate on the dev set
# y_dev_pred = clf.predict(X_dev)
# accuracy_dev = accuracy_score(y_dev, y_dev_pred)
# print(f"Accuracy on dev set: {accuracy_dev * 100:.2f}%")

# # Perform feature selection using SelectFromModel on the train set
# sfm = SelectFromModel(clf, threshold='median')
# sfm.fit(X_train, y_train)

# # Transform the features based on the selected features
# X_train_selected = sfm.transform(X_train)
# X_dev_selected = sfm.transform(X_dev)

# # Train the classifier on the selected features
# clf.fit(X_train_selected, y_train)

# # Evaluate on the dev set with selected features
# y_dev_pred_selected = clf.predict(X_dev_selected)
# accuracy_dev_selected = accuracy_score(y_dev, y_dev_pred_selected)
# print(f"Accuracy on dev set with selected features: {accuracy_dev_selected * 100:.2f}%")

# # Finally, evaluate on the test set with selected features
# X_test_selected = sfm.transform(X_test)
# y_test_pred = clf.predict(X_test_selected)
# accuracy_test = accuracy_score(y_test, y_test_pred)
# print(f"Accuracy on test set with selected features: {accuracy_test * 100:.2f}%")

# -----------------------------------------------------
# PHQ Resgression, feature importance

X_train = minimal_features_train
y_train = labels_train_df['PHQ_Score']

X_dev = minimal_features_dev
y_dev = labels_dev_df['PHQ_Score']

X_test = minimal_features_test
y_test = labels_test_df['PHQ_Score']

# Train a RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Perform feature selection using SelectFromModel
sfm = SelectFromModel(regressor, threshold='median')
sfm.fit(X_train, y_train)

# Transform the features based on the selected features
X_train_selected = sfm.transform(X_train)
X_dev_selected = sfm.transform(X_dev)
X_test_selected = sfm.transform(X_test)

# Train the regressor on the selected features
regressor.fit(X_train_selected, y_train)

# Get feature importances from the trained regressor
feature_importances = regressor.feature_importances_

# Get indices of features sorted by importance
indices_selected = np.argsort(feature_importances)[::-1]

# Print feature importances
print("Feature Importances for Selected Features:")
for f in range(X_train_selected.shape[1]):
    print(f"Feature {f + 1}: Index {indices_selected[f]}, Importance: {feature_importances[indices_selected[f]]}")

# Assuming X_train_selected is your selected feature matrix
selected_feature_indices = indices_selected

# Make sure the indices are within the range of the actual number of features
selected_feature_indices = selected_feature_indices[selected_feature_indices < len(df_openface.columns)]

# Get the names of selected features
selected_feature_names = df_openface.columns[selected_feature_indices]

# Print or use the selected feature names
print("Selected Feature Names:")
print(selected_feature_names)

# Print or use the selected feature names with direction of impact
print("Selected Feature Names with Direction of Impact:")
for f in range(len(selected_feature_indices)):
    index = selected_feature_indices[f]
    importance = feature_importances[index]
    direction = "Positive" if importance > 0 else "Negative"
    print(f"Feature {f + 1}: Index {index}, Importance: {importance:.4f}, Impact Direction: {direction}")

# Predict on the dev set
y_dev_pred_selected = regressor.predict(X_dev_selected)
print('y_dev: ', y_dev)
print('y_dev_pred_selected: ', y_dev_pred_selected)

mae_dev_selected = mean_absolute_error(y_dev, y_dev_pred_selected)
rmse_dev_selected = np.sqrt(mean_squared_error(y_dev, y_dev_pred_selected))
print(f"MAE on dev set with selected features: {mae_dev_selected:.2f}")
print(f"RMSE on dev set with selected features: {rmse_dev_selected:.2f}")

# Predict on the test set
y_test_pred_selected = regressor.predict(X_test_selected)
print('y_test: ', y_test)
print('y_test_pred_selected: ', y_test_pred_selected)

mae_test_selected = mean_absolute_error(y_test, y_test_pred_selected)
rmse_test_selected = np.sqrt(mean_squared_error(y_test, y_test_pred_selected))
print(f"MAE on test set with selected features: {mae_test_selected:.2f}")
print(f"RMSE on test set with selected features: {rmse_test_selected:.2f}")



# ------------------------------------------------------------------------------------
# # Feature selection as part of a pipeline and nested cross-validation

# # Create a pipeline with feature selection and regression steps
# regressor = RandomForestRegressor(random_state=42)

# clf = Pipeline([
#     ('feature_selection', SelectFromModel(LinearSVR(dual="auto", C=0.01))),
#     ('regression', regressor)
# ])

# # Define the parameter grid for hyperparameter tuning
# param_grid = {
#     'feature_selection__estimator__C': [0.01, 0.1, 1, 10],
#     'regression__n_estimators': [50, 100, 200],
#     'regression__max_depth': [None, 10, 20],
# }

# # Define a scorer (e.g., mean absolute error) for the grid search
# scorer = make_scorer(mean_absolute_error)

# # Create the outer cross-validation
# outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Create the inner cross-validation
# inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# # Create the grid search with the pipeline and parameter grid
# grid_search = GridSearchCV(clf, param_grid, scoring=scorer, cv=inner_cv)

# # Perform nested cross-validation using the training set and development set
# nested_predictions = cross_val_predict(grid_search, pd.concat([minimal_features_train, minimal_features_dev]), pd.concat([y_train, y_dev]), cv=outer_cv)

# # Fit the pipeline on the entire training set and development set using the best hyperparameters
# grid_search.fit(pd.concat([minimal_features_train, minimal_features_dev]), pd.concat([y_train, y_dev]))

# # Make predictions on the test set
# y_test_pred = grid_search.predict(minimal_features_test)

# # Calculate MAE and RMSE on the development set
# mae_dev = mean_absolute_error(pd.concat([y_train, y_dev]), nested_predictions)
# rmse_dev = np.sqrt(mean_squared_error(pd.concat([y_train, y_dev]), nested_predictions))
# print(f"Nested Cross-Validation Mean Absolute Error on Development Set: {mae_dev:.2f}")
# print(f"Nested Cross-Validation Root Mean Squared Error on Development Set: {rmse_dev:.2f}")

# # Calculate MAE and RMSE on the test set
# mae_test = mean_absolute_error(y_test, y_test_pred)
# rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
# print(f"Test Set Mean Absolute Error: {mae_test:.2f}")
# print(f"Test Set Root Mean Squared Error: {rmse_test:.2f}")
# print(f"Best Hyperparameters: {grid_search.best_params_}")


# ------------------------------------------------------------------------------------

# Extract minimal features using tsfresh
# # Assuming 'PHQ_Binary' is the target column for prediction
# target_column = merged_df['PHQ_Binary']
# merged_df = merged_df.drop(['PHQ_Binary'], axis=1)


# minimal_features = extract_features(merged_df, column_id='id', column_sort='frame', default_fc_parameters=MinimalFCParameters())
# # minimal_features = extract_features(merged_df, column_id='id', column_sort='frame', default_fc_parameters=EfficientFCParameters())
# # Impute the result (fill NaN values)
# minimal_features = impute(minimal_features)
# print('minimal_features: ', minimal_features)
# minimal_features.to_csv('./minimal_features.csv', index=False)

# # # Select relevant features based on significance tests
# # selected_features = select_features(minimal_features, target_column)

# # # Review the selected features
# # print('selected_features: ', selected_features)


# X_train = extract_features(merged_df[merged_df['id'].isin(labels_train_df['id'])], 
#                            column_id='id', column_sort='frame', default_fc_parameters=MinimalFCParameters())
# y_train = labels_train_df['PHQ_Binary']



# Feature Selection using Tsfresh 
# AU_intensities = df_openface.filter(like='_r').columns.to_list() 
# AU_presence = df_openface.filter(like='_c').columns.to_list() 
# eye_gaze = df_openface.filter(like='gaze').columns.to_list()
# head_pose = df_openface.filter(like='pose').columns.to_list()

# print("AU_intensities: ", AU_intensities)
# print("AU_presence: ", AU_presence)
# print("eye_gaze: ", eye_gaze)
# print("head_pose: ", head_pose)

# df_head_pose = df_openface.drop(AU_intensities+eye_gaze+AU_presence, axis=1)
# print("df_head_pose: ", df_head_pose)



# extracted_features_headpose = extract_features(df_head_pose, column_id="id", column_sort="frame")
# impute(extracted_features_headpose)
# features_filtered_headpose = select_features(extracted_features_headpose, y)

# print('extracted_features_headpose: ', extracted_features_headpose)
# print("--------------------------------------")
# print('features_filtered_headpose: ', features_filtered_headpose)
# print("--------------------------------------")
# print('y: ', y)

# def get_data_openface(train_val_test_dir):
#     participants_IDs_path = {}
#     df_openface = pd.DataFrame()
                               
#     for name in os.listdir(splitted_dataset_dir + train_val_test_dir):
#         if name.startswith('.'):
#             continue
#         participants_IDs_path[(name[:3])] = splitted_dataset_dir+train_val_test_dir+'/'+name
    
#     for key, value in participants_IDs_path.items():
#         df_temp = pd.read_csv(value + '/features/' + f'{key}_OpenFace2.1.0_Pose_gaze_AUs.csv')
#         indices_to_remove = df_temp[df_temp['confidence'] < 80].index
#         print('indices_to_remove: ', indices_to_remove)
#         print('indices_to_remove len: ', len(indices_to_remove))
#         # indices_to_remove = df_temp[df_temp['success'] == 0].index

#         df_temp = df_temp.drop(index=indices_to_remove)
#         df_temp.insert(0, 'ID', key)
#         df_openface = pd.concat([df_openface, df_temp])
#     return df_openface

# df_openface_train = get_data_openface('train_data')
# df_openface_dev = get_data_openface('dev_data')
# df_openface_test = get_data_openface('test_data')

# df_openface_train = df_openface_train.sort_values(by=['ID','frame'])
# df_openface_dev = df_openface_dev.sort_values(by=['ID','frame'])
# df_openface_test = df_openface_test.sort_values(by=['ID','frame'])

# print('df_openface_train head: ', df_openface_train.head())

# df_openface_train = df_openface_train.drop(['timestamp', 'confidence', 'success'], axis=1)
# df_openface_dev = df_openface_dev.drop(['timestamp', 'confidence', 'success'], axis=1)
# df_openface_test = df_openface_test.drop(['timestamp', 'confidence', 'success'], axis=1)

# df_openface_train = df_openface_train.reset_index(drop=True)
# df_openface_dev = df_openface_dev.reset_index(drop=True)
# df_openface_test = df_openface_test.reset_index(drop=True)

# df_openface_all_sets = pd.concat([df_openface_train, df_openface_dev, df_openface_test])
# df_openface_all_sets = df_openface_all_sets.reset_index(drop=True)

# print('df_openface_all_sets head: ', df_openface_all_sets.head())

# # --------------------------------------------------------------------------------------------------------------
# # Extracting features using tsfresh

# AU_intensities = df_openface_all_sets.filter(like='_r').columns.to_list() 
# AU_presence = df_openface_all_sets.filter(like='_c').columns.to_list() 
# eye_gaze = df_openface_all_sets.filter(like='eye').columns.to_list()
# head_pose = df_openface_all_sets.filter(like='head').columns.to_list()

# df_AU_intensities = df_openface_all_sets.drop(AU_presence+eye_gaze+head_pose, axis=1)
# df_AU_presence = df_openface_all_sets.drop(AU_intensities+eye_gaze+head_pose, axis=1)
# df_eye_gaze = df_openface_all_sets.drop(AU_intensities+AU_presence+head_pose, axis=1)
# df_head_pose = df_openface_all_sets.drop(AU_intensities+eye_gaze+AU_presence, axis=1)

# # Head Pose features
# extracted_features_headpose = extract_features(df_head_pose, column_id="ID", column_sort="frame")
# impute(extracted_features_headpose)
# features_filtered_headpose = select_features(extracted_features_headpose, y)

# print('extracted_features_headpose: ', extracted_features_headpose)
# print('features_filtered_headpose: ', features_filtered_headpose)
# print('y: ', y)

# # Eye Gaze features
# extracted_features_eyegaze= extract_features(df_eye_gaze, column_id="ID", column_sort="frame")
# impute(extracted_features_eyegaze)
# features_filtered_eyegaze = select_features(extracted_features_eyegaze, y)

# print('extracted_features_eyegaze: ', extracted_features_eyegaze)
# print('features_filtered_eyegaze: ', features_filtered_eyegaze)
# print('y: ', y)
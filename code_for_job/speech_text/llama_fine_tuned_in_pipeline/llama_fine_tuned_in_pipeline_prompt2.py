import os
import re
import glob
import joblib
import openai
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import StackingRegressor
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AdamW
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer
import sys
from dotenv import load_dotenv

sys.path.append('/home/hpc/empk/empk004h/depression-detection/code_for_job/speech_text/llama_fine_tuned_in_pipeline')

# Import the other file
import finetuned_llama_inference

# Load environment variables from .env file
load_dotenv()

os.environ['CURL_CA_BUNDLE'] = ''

# SECURELY LOAD TOKEN FROM ENVIRONMENT
token = os.getenv("HUGGING_FACE_TOKEN")
if not token:
    raise ValueError("Hugging Face token not found. Make sure to set HUGGING_FACE_TOKEN in your .env file.")

directory_path = '/home/hpc/empk/empk004h/depression-detection/data/transcripts_from_whisper/'
transcripts_df = pd.DataFrame(columns=["id", "text"])

# ------------------------------------------------------------------------------------------------
# Extract IDs from filenames and text from files
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_id = filename[:3]
        with open(os.path.join(directory_path, filename), "r") as file:
            file_contents = file.read()
        transcripts_df = transcripts_df._append({"id": file_id, "text": file_contents}, ignore_index=True)

transcripts_df["id"] = transcripts_df["id"].astype("int64")

labels_dev_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/dev_split.csv')
labels_dev_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_train_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/train_split.csv')
labels_train_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_test_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/test_split.csv')
labels_test_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']
# ------------------------------------------------------------------------------------------------
# Merge dataframes on ID
df_train = pd.merge(labels_train_df, transcripts_df, on='id')
df_dev = pd.merge(labels_dev_df, transcripts_df, on='id')
df_test = pd.merge(labels_test_df, transcripts_df, on='id')

df_dev = df_dev.sort_values(by="id")
df_train = df_train.sort_values(by="id")
df_test = df_test.sort_values(by="id")

print('def_dev: ', df_dev.head())
print('max PHQ score in df_train: ', max(df_train['PHQ_Score']))
# ------------------------------------------------------------------------------------------------
prompt = """ Your task is to read the following text which is an interview with a person and to summarize
        the key points that might be related to the depression of the person. Be concise and to the point. 
        It is very essential that you write your answer in the first-person perspective, as if the interviewee 
        is narrating about himself or herself. """
# ------------------------------------------------------------------------------------------------
# Reading from the CSV files so we don't have to send requests over and over to gpt

# Define the input directory path
input_directory = "/home/hpc/empk/empk004h/depression-detection/notebooks/prompts_outputs/"
prompt_number = 2

# Define the folder path for the prompt number
prompt_directory = os.path.join(input_directory, str(prompt_number))

# Read the CSV files for each dataframe
dev_filepath = os.path.join(prompt_directory, f"dev_prompt{prompt_number}.csv")
train_filepath = os.path.join(prompt_directory, f"train_prompt{prompt_number}.csv")
test_filepath = os.path.join(prompt_directory, f"test_prompt{prompt_number}.csv")

# Read the dataframes from CSV files
df_dev = pd.read_csv(dev_filepath)
df_train = pd.read_csv(train_filepath)
df_test = pd.read_csv(test_filepath)
# ------------------------------------------------------------------------------------------------  
print('prompt_number: ', prompt_number)
print('prompt: ', prompt)
# ------------------------------------------------------------------------------------------------
# Now you can use the fine-tuned Llama model for inference
def extract_features(df_train, df_dev, df_test):

    # Extracting texts from dataframes
    texts_train = df_train['completions']
    texts_dev = df_dev['completions']
    texts_test = df_test['completions']

    # Use llama_inference to get features
    X_train_features = finetuned_llama_inference.run_inference(texts_train)
    X_dev_features = finetuned_llama_inference.run_inference(texts_dev)
    X_test_features = finetuned_llama_inference.run_inference(texts_test)

    print('len X_train_features: ', len(X_train_features))
    print('len X_dev_features: ', len(X_dev_features))
    print('X_test_features: ', len(X_test_features))

    return X_train_features, X_dev_features, X_test_features

X_train_features, X_dev_features, X_test_features = extract_features(df_train, df_dev, df_test)

# ------------------------------------------------------------------------------------------------
# Training an SVR model

fine_tuned_features_train = np.array(X_train_features)
fine_tuned_features_dev = np.array(X_dev_features)
fine_tuned_features_test = np.array(X_test_features)

print('train shape: ', np.shape(fine_tuned_features_train))
print('dev shape: ', np.shape(fine_tuned_features_dev))
print('test shape: ', np.shape(fine_tuned_features_test))
  
X_train = fine_tuned_features_train
X_dev = fine_tuned_features_dev
X_test = fine_tuned_features_test

y_train = np.array(df_train['PHQ_Score'])
y_dev = np.array(df_dev['PHQ_Score'])
y_test = np.array(df_test['PHQ_Score'])
    
# # Define the parameter grid for GridSearchCV
# param_grid = {
#     'kernel': ['linear', 'rbf', 'poly'],
#     'C': [0.01, 0.1, 1, 2, 5, 7, 10, 100],
#     'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
#     'coef0': [0.0, 0.5, 1.0, 2.0, 3.0],
#     'degree': [2, 3, 4]
# }

# # Perform grid search to find the best parameters
# grid_search = GridSearchCV(SVR(), param_grid, cv=5)
# grid_search.fit(X_dev, y_dev)

# # Print the best parameters chosen by GridSearchCV
# print("Best parameters found:")
# print(grid_search.best_params_)

# # Get the best SVR model from the grid search
# svr = grid_search.best_estimator_

# Define fixed parameters
svr_params = {
    'kernel': 'rbf',  # Fixed kernel
    'C': 1.0,         # Fixed regularization parameter
    'gamma': 'scale', # Fixed kernel coefficient
    'coef0': 0.0,     # Fixed independent term in kernel function
    'degree': 3       # Fixed degree of the polynomial kernel function
}

# Instantiate SVR with fixed parameters
svr = SVR(**svr_params)

# Train the SVR model on X_train and y_train
svr.fit(X_train, y_train)

# Predict on X_train and calculate the mean squared error and mean absolute error
y_pred_train = svr.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
print('RMSE for train: ', np.sqrt(mse_train))
print('MAE for train: ', mae_train)

print('SRV results based on fine-tuned Llama model: ')

# Print predicted and true values for train set
print("Train set:")
for true, pred in zip(y_train, y_pred_train):
    print("True:", true, "\tPredicted:", pred)

# Predict on X_dev and calculate the mean squared error and mean absolute error
y_pred_dev = svr.predict(X_dev)
mse_dev = mean_squared_error(y_dev, y_pred_dev)
mae_dev = mean_absolute_error(y_dev, y_pred_dev)
print('RMSE for dev: ', np.sqrt(mse_dev))
print('MAE for dev: ', mae_dev)

# Print predicted and true values for dev set
print("Dev set:")
for true, pred in zip(y_dev, y_pred_dev):
    print("True:", true, "\tPredicted:", pred)

# Predict on X_test and calculate the mean squared error and mean absolute error
y_pred_test = svr.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
print('RMSE for test: ', np.sqrt(mse_test))
print('MAE for test: ', mae_test)

# Print predicted and true values for test set
print("Test set:")
for true, pred in zip(y_test, y_pred_test):
    print("True:", true, "\tPredicted:", pred)
# ------------------------------------------------------------------------------------------------
# Nested Cross-Validation (CV)

# Combine X_train and X_dev for the outer loop of nested cross-validation
X_train_dev = np.concatenate((X_train, X_dev), axis=0)
y_train_dev = np.concatenate((y_train, y_dev), axis=0)

# Define the parameter grid for GridSearchCV
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.01, 0.1, 1, 2, 5, 7, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    'coef0': [0.0, 0.5, 1.0, 2.0, 3.0],
    'degree': [2, 3, 4]
}

# Outer loop: Nested Cross-Validation
outer_cv = 5  # Number of folds for outer cross-validation
inner_cv = 3  # Number of folds for inner cross-validation

best_scores = []

for i, (train_index, dev_index) in enumerate(KFold(n_splits=outer_cv, shuffle=True).split(X_train_dev)):
    X_train_fold, X_dev_fold = X_train_dev[train_index], X_train_dev[dev_index]
    y_train_fold, y_dev_fold = y_train_dev[train_index], y_train_dev[dev_index]

    grid_search = GridSearchCV(SVR(), param_grid, cv=inner_cv)
    grid_search.fit(X_train_fold, y_train_fold)

    best_scores.append(grid_search.best_score_)

    print("Outer Fold:", i+1)
    print("Best parameters:", grid_search.best_params_)

    # Model performance on dev set
    best_model = grid_search.best_estimator_
    y_pred_dev = best_model.predict(X_dev_fold)
    mse_dev = mean_squared_error(y_dev_fold, y_pred_dev)
    mae_dev = mean_absolute_error(y_dev_fold, y_pred_dev)
    print("Performance on dev set:")
    print('RMSE for dev: ', np.sqrt(mse_dev))
    print('MAE for dev: ', mae_dev)

print("Mean CV Score:", np.mean(best_scores))

# ------------------------------------------------------------------------------------------------
# Batch Normalization

# Normalize input data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_dev_normalized = scaler.transform(X_dev)
X_test_normalized = scaler.transform(X_test)

# Train the SVR model on normalized data
svr.fit(X_train_normalized, y_train)

print('Results with Batch Normalization: ')
# Evaluate the model
y_pred_train = svr.predict(X_train_normalized)
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
print('RMSE for train: ', np.sqrt(mse_train))
print('MAE for train: ', mae_train)

# Predict on dev set
y_pred_dev = svr.predict(X_dev_normalized)
mse_dev = mean_squared_error(y_dev, y_pred_dev)
mae_dev = mean_absolute_error(y_dev, y_pred_dev)
print('RMSE for dev: ', np.sqrt(mse_dev))
print('MAE for dev: ', mae_dev)

# Predict on test set
y_pred_test = svr.predict(X_test_normalized)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
print('RMSE for test: ', np.sqrt(mse_test))
print('MAE for test: ', mae_test)



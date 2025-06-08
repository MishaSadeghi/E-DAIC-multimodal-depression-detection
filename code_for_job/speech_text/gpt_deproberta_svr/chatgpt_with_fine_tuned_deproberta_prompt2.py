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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingRegressor
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AdamW
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer

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
# !!! These are the prompts that we will use for gpt-3.5-turbo API !!!
# keep that in mind that you are now using the prompt2 outputs from the previous model which was gpt3.5-turbo 
# because they were better than the new model regarding being in the first person narrating 
prompt = """ Your task is to read the following text which is an interview with a person and to summarize the key points that might be related to the depression of the person. Be concise and to the point. It is very essential that you write your answer in the first-person perspective, as if the interviewee is narrating about himself or herself. """

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
# It's for creating the results list from result dictionaries after reading from the CSV files

print('prompt_number: ', prompt_number)
print('prompt: ', prompt)
# ------------------------------------------------------------------------------------------------
# Feature extraction based on fine-tuned deproberta
# Extracting deproberta features (probabilities) from completions for all of the prompts results

tokenizer = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")

def extract_features(df_train, df_dev, df_test, model):
    for df in [df_train, df_dev, df_test]:
        # Remove newlines and non-meaningful characters
        df['completions'] = df['completions'].replace(r'\n', ' ', regex=True)  # Replace newlines with spaces
        df['completions'] = df['completions'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)  # Remove non-alphanumeric and non-space characters
        df['completions'] = df['completions'].replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with a single space

    X_train = df_train['completions']
    X_dev = df_dev['completions']
    X_test = df_test['completions']
    
    print(len(X_train))
    print(len(X_dev))
    print(len(X_test))

    # Extract features from train data
    X_train_features = []
    for i in range(len(X_train)):
        input_ids = torch.tensor(tokenizer.encode(X_train[i], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]
        probs = torch.softmax(logits, dim=1)
        predicted_label_index = torch.argmax(probs, dim=1).item()
        X_train_features.append(probs.detach().numpy())
        print('train i: ', i)

    # Extract features from dev data
    X_dev_features = []
    for i in range(len(X_dev)):
        input_ids = torch.tensor(tokenizer.encode(X_dev[i], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]
        probs = torch.softmax(logits, dim=1)
        predicted_label_index = torch.argmax(probs, dim=1).item()
        X_dev_features.append(probs.detach().numpy())
        print('dev i: ', i)

    # Extract features from test data
    X_test_features = []
    for i in range(len(X_test)):
        input_ids = torch.tensor(tokenizer.encode(X_test[i], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]
        probs = torch.softmax(logits, dim=1)
        predicted_label_index = torch.argmax(probs, dim=1).item()
        X_test_features.append(probs.detach().numpy())
        print('test i: ', i)

    return (X_train_features, X_dev_features, X_test_features)


latest_model_file = "/home/hpc/empk/empk004h/depression-detection/model/best_deproberta_finetuned_train_3rd_phq_range_12_ep_lr_5e-6"

# Load the model 
model = AutoModelForSequenceClassification.from_pretrained(latest_model_file)

X_train_features, X_dev_features, X_test_features = extract_features(df_train, df_dev, df_test, model)
# ------------------------------------------------------------------------------------------------
# Training and Evaluating and Testing

y_train = np.array(df_train['PHQ_Score'])
y_dev = np.array(df_dev['PHQ_Score'])
y_test = np.array(df_test['PHQ_Score'])
    
deproberta_features_train = np.array(X_train_features)
deproberta_features_dev = np.array(X_dev_features)
deproberta_features_test = np.array(X_test_features)

print('train shape: ', np.shape(deproberta_features_train))
print('dev shape: ', np.shape(deproberta_features_dev))
print('test shape: ', np.shape(deproberta_features_test))

# Reshape the features
deproberta_features_train = deproberta_features_train.reshape(deproberta_features_train.shape[0], deproberta_features_train.shape[2])
deproberta_features_dev = deproberta_features_dev.reshape(deproberta_features_dev.shape[0], deproberta_features_dev.shape[2])
deproberta_features_test = deproberta_features_test.reshape(deproberta_features_test.shape[0], deproberta_features_test.shape[2])

X_train = deproberta_features_train
X_dev = deproberta_features_dev
X_test = deproberta_features_test

# Define the parameter grid for GridSearchCV
param_grid = {
    'kernel': ['rbf', 'poly'],
    'C': [0.01, 0.1, 10],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 1.0, 2.0],
    'degree': [2, 3]
}

# Perform grid search to find the best parameters
grid_search = GridSearchCV(SVR(), param_grid, cv=5)
grid_search.fit(X_dev, y_dev)

# Get the best SVR model from the grid search
svr = grid_search.best_estimator_

# Train the SVR model on X_train and y_train
svr.fit(X_train, y_train)

# Predict on X_train and calculate the mean squared error and mean absolute error
y_pred_train = svr.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
print('RMSE for train: ', np.sqrt(mse_train))
print('MAE for train: ', mae_train)

print('SRV results based on fine-tuned DepRoberta model: ')

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





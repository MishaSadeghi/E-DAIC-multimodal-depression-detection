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

# setting a different cache dir for hugging_face
# os.environ['HF_HOME'] = '/home/woody/empk/empk004h/huggingface_cache/.cache'

directory_path = '/home/hpc/empk/empk004h/depression-detection/data/transcripts_from_whisper/'
transcripts_df = pd.DataFrame(columns=["id", "text"])
# ------------------------------------------------------------------------------------------------
# Extract IDs from filenames and text from files
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_id = filename[:3]
        with open(os.path.join(directory_path, filename), "r") as file:
            file_contents = file.read()
        transcripts_df = transcripts_df.append({"id": file_id, "text": file_contents}, ignore_index=True)

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
prompt1 = """ Your task is to read the following text which is an interview with a person and to summarize the key points that might be related to the depression of the person. Be concise and to the point.""" 
# keep that in mind that you are now using the prompt2 outputs from the previous model which was gpt3.5-turbo 
# because they were better than the new model regarding being in the first person narrating 
prompt2 = """ Your task is to read the following text which is an interview with a person and to summarize the key points that might be related to the depression of the person. Be concise and to the point. It is very essential that you write your answer in the first-person perspective, as if the interviewee is narrating about himself or herself. """
prompt3 = """ After reading the interview, briefly summarize the main aspects that pertain to the person's depression. """
prompt4 = """ Based on the interview, highlight the key factors that might be indicative of the interviewee's depression. """
prompt5 = """ Your task is to summarize the interviewee's main points that could be linked to their depression. Keep it concise. """
prompt6 = """ After reading the interview, identify and summarize the main challenges or difficulties the interviewee faces that are indicative of depression. """
prompt7 = """ Based on the interview, provide a concise analysis of the interviewee's emotional state and behaviors that may indicate the presence of depression. """
prompt8 = """ Read the interview carefully and extract the most significant indicators of depression exhibited by the interviewee. Summarize them concisely. """
prompt9 = """ Your task is to analyze the interviewee's responses and highlight the key signs or symptoms of depression that are evident in the interview. """
prompt10 = """ Provide a brief summary of the interview, focusing on aspects that strongly suggest the presence of depression in the interviewee. """

prompts = {str(i): eval(f"prompt{i}") for i in range(1, 11)}
# ------------------------------------------------------------------------------------------------
# Reading from the CSV files so we don't have to send requests over and over to gpt

# Define the input directory path
input_directory = "/home/hpc/empk/empk004h/depression-detection/notebooks/prompts_outputs/"
prompt_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Iterate through the results
for prompt_number in prompt_numbers:
    # prompt_number = result['prompt_number']

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

    # Rename the dataframes using exec()
    exec(f"df_dev_{prompt_number} = df_dev")
    exec(f"df_train_{prompt_number} = df_train")
    exec(f"df_test_{prompt_number} = df_test")
# ------------------------------------------------------------------------------------------------
# It's for creating the results list from result dictionaries after reading from the CSV files

# Define a list to store the results
results = []

for prompt_number, prompt in prompts.items():
    
    print('prompt_number: ', prompt_number)
    print('prompt: ', prompt)
    
    # Get the dataframe for the current prompt number
    df_dev = eval(f"df_dev_{prompt_number}")
    df_train = eval(f"df_train_{prompt_number}")
    df_test = eval(f"df_test_{prompt_number}")
    
    # Store the results in a dictionary
    result = {
        'prompt_number': prompt_number,
        'prompt': prompt,
        'df_dev': df_dev,
        'df_train': df_train,
        'df_test': df_test
    }
    
    # Append the result to the list
    results.append(result)
# ------------------------------------------------------------------------------------------------
# First step for fine tuning the Deproberta model is to create labels similar to the Deproberta model labels

# Define a function to map PHQ scores to categories
def map_phq_score_to_category(score):
    if score >= 14:
        # "severe" group
        return 0
    elif score >= 7 and score <= 13:
        # "moderate" group
        return 1
    elif score < 7:
        # "not depression" group
        return 2

# Loop through each dictionary in the results list
for result in results:
    # Get the dataframe for the current prompt number
    df_train = result['df_train']
    df_dev = result['df_dev']
    df_test = result['df_test']
    
    # Add a new column based on the PHQ scores
    df_train["PHQ_Group"] = df_train["PHQ_Score"].apply(map_phq_score_to_category)
    df_dev["PHQ_Group"] = df_dev["PHQ_Score"].apply(map_phq_score_to_category)
    df_test["PHQ_Group"] = df_test["PHQ_Score"].apply(map_phq_score_to_category)
# ------------------------------------------------------------------------------------------------
# Fine tuning the DepRoberta model

model_dir = "/home/hpc/empk/empk004h/depression-detection/model/ensemble/"

# Define a function for fine-tuning the model on a specific training set
def fine_tune_model(train_texts, train_labels, dev_texts, dev_labels, max_epochs=20):
    # Prepare the data for the current training set
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), 
                                  torch.tensor(train_encodings['attention_mask']),
                                  torch.tensor(train_labels))

    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)
    dev_dataset = TensorDataset(torch.tensor(dev_encodings['input_ids']), 
                                torch.tensor(dev_encodings['attention_mask']),
                                torch.tensor(dev_labels))
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

    # Define optimizer and other fine-tuning parameters here
    # optim = torch.optim.AdamW(model.parameters(), lr=5e-6) #=> 68% accuracy on dev & 61% accuracy on test
    # optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # optim = torch.optim.AdamW(model.classifier.parameters(), lr=5e-6) # for fine-tuning only the final layer 
    # optim = torch.optim.AdamW(model.classifier.out_proj.parameters(), lr=5e-6) # for fine-tuning last 2 fully connected layers

    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # for param in model.classifier.dense.parameters():   # for fine-tuning last 2 fully connected layers
    #     param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if 'classifier' not in name:  # Unfreeze classifier layer
    #         param.requires_grad = False


    # Unfreeze the last 4 layers of the base model's encoder
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'encoder.layer' in name:
            layer_num = int(name.split('encoder.layer.')[-1].split('.')[0])
            if layer_num >= (model.config.num_hidden_layers - 4):  # Unfreeze last 4 layers
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    optim = torch.optim.AdamW(
    [
        {'params': model.classifier.parameters()},
        {'params': model.base_model.encoder.layer[-10:].parameters(), 'lr': 1e-5}  # Fine-tuning last 4 layers
    ],
    lr=5e-6
    )
    
    # Define the scheduler
        # scheduler = CosineAnnealingLR(optim, T_max=5, eta_min=1e-7)

    # Parameters for early stopping
    best_dev_loss = float('inf')  # Track the best development loss
    best_epoch = -1  # Track the epoch with the best development loss
    epochs_since_best_loss = 0  # Count the number of epochs since the best loss was updated
    max_epochs_without_improvement = 3  # Number of epochs without improvement to trigger early stopping

    # Train the model
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Start fine-tuning
    for epoch in range(max_epochs):
        model.train() 
        print('len train_loader: ', len(train_loader))
        for step, batch in enumerate(train_loader):
            optim.zero_grad()
            input_ids, attn_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            # scheduler.step()
            
            current_lr = optim.param_groups[0]['lr']
            print(f"Epoch {epoch}, Batch {step}, Learning Rate: {current_lr:.8f}, Train Loss: {loss.item():.3f}")
            
            # if step % 100 == 0:
            #     print(f"Epoch {epoch}, Batch {step}, Train Loss: {loss.item():.3f}")
            
        # Calculate dev loss after each epoch
        model.eval()
        with torch.no_grad():
            dev_loss = 0
            dev_total = 0
            for dev_batch in dev_loader:
                dev_input_ids, dev_attn_mask, dev_labels = tuple(t.to(device) for t in dev_batch)
                dev_outputs = model(dev_input_ids, attention_mask=dev_attn_mask, labels=dev_labels)
                dev_loss += dev_outputs.loss.item() * dev_labels.size(0)
                dev_total += dev_labels.size(0)
            dev_avg_loss = dev_loss / dev_total
            print(f"Epoch {epoch}, Batch {step}, Dev Loss: {dev_avg_loss:.3f}")
            
            # Check for early stopping
            if dev_avg_loss < best_dev_loss:
                best_dev_loss = dev_avg_loss
                best_epoch = epoch  # Update the best epoch
                print('best_epoch: ', best_epoch)
                epochs_since_best_loss = 0

                # Update the best model's state dict
                best_model_state_dict = model.state_dict()
                
                # Save the model at the best epoch
                model_name = f"fine_tuned_model_prompt_{result['prompt_number']}_best_epoch_{best_epoch}"
                model.save_pretrained(model_dir + model_name)
                print("The best model has been saved!")
            else:
                epochs_since_best_loss += 1
                
        print('epochs_since_best_loss: ', epochs_since_best_loss)
#         if epochs_since_best_loss >= max_epochs_without_improvement:
#             print("Early stopping triggered. No improvement in dev loss.")
#             break

    return model

# # Fine-tune the model for each training set in the results list
# for result in results:
#     # Load the tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")
#     model = AutoModelForSequenceClassification.from_pretrained("rafalposwiata/deproberta-large-depression", num_labels=3)

#     # Set up device (GPU or CPU)
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model.to(device)

#     best_model_state_dict = None

#     print('prompt_number: ', result['prompt_number'])
#     df_train = result['df_train']
#     df_dev = result['df_dev']
#     train_texts = df_train['completions'].tolist()
#     train_labels = np.array(df_train['PHQ_Group'].tolist())
#     dev_texts = df_dev['completions'].tolist()
#     dev_labels = np.array(df_dev['PHQ_Group'].tolist())
    
#     # Fine-tune the model on the current training set
#     model = fine_tune_model(train_texts, train_labels, dev_texts, dev_labels)
#     print('best_model_state_dict: ', best_model_state_dict)
#     if best_model_state_dict is not None:
#         # Save the fine-tuned model
#         model_name = f"fine_tuned_model_prompt_{result['prompt_number']}"
#         model.load_state_dict(best_model_state_dict)
#         model.save_pretrained(model_dir + model_name)
#         print("The model has been saved!")

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

# Function to extract the epoch number from the model file name
def get_epoch_number(file_path):
    return int(file_path.split("_")[-1])

# List to store features for all dataframes
all_features = []
# Iterate through each prompt result
for result in results:
    prompt_number = result['prompt_number']
    print('Extracting Features for Prompt ', prompt_number, ' ..... ')

    df_train = result['df_train']
    df_dev = result['df_dev']
    df_test = result['df_test']

    y_train = np.array(df_train['PHQ_Score'])
    y_dev = np.array(df_dev['PHQ_Score'])
    y_test = np.array(df_test['PHQ_Score'])

    # Find the corresponding model file based on the prompt number
    # model_files = glob.glob(f"{model_dir}/fine_tuned_model_prompt_{prompt_number}_best_epoch_*")


    # # Ensure that there are model files for the current prompt
    # if not model_files:
    #     raise ValueError(f"No model files found for prompt {prompt_number}")

    # # Select the model with the greatest epoch
    # latest_model_file = max(model_files, key=get_epoch_number)
    # print('latest_model_file: ', latest_model_file)

    latest_model_file = "best_deproberta_finetuned_train_3rd_phq_range_12_ep_lr_5e-6"

    # Load the model 
    model = AutoModelForSequenceClassification.from_pretrained(latest_model_file)

    X_train_features, X_dev_features, X_test_features = extract_features(df_train, df_dev, df_test, model)
    
    # Store features in a dictionary
    prompt_features = {
        'prompt_number': prompt_number,
        'X_train_features': X_train_features,
        'X_dev_features': X_dev_features,
        'X_test_features': X_test_features,
        'y_train': y_train,
        'y_dev': y_dev,
        'y_test': y_test
    }

    # Append the prompt features to the all_features list
    all_features.append(prompt_features)

    print('Succefullly added prompt_features for prompt_number ', prompt_number)
    print('---------------------------------------------------')
# ------------------------------------------------------------------------------------------------
# Training and Evaluating and Testing

# List to store SVR models and their results for each prompt
svr_models_results = []

# List to store predictions for each prompt
all_predictions_dev = []
all_predictions_test = []

# Iterate through each prompt result
for prompt_features in all_features:
    prompt_number = prompt_features['prompt_number']
    X_train_features = np.array(prompt_features['X_train_features'])
    X_dev_features = np.array(prompt_features['X_dev_features'])
    X_test_features = np.array(prompt_features['X_test_features'])
    y_train = prompt_features['y_train']
    y_dev = prompt_features['y_dev']
    y_test = prompt_features['y_test']
    
    print(f'Training based on Prompt {prompt_number}')

    # Convert to numpy arrays
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

    # Normalize X_train, X_dev, and X_test
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(deproberta_features_train)
#     X_dev = scaler.transform(deproberta_features_dev)
#     X_test = scaler.transform(deproberta_features_test)
    
    X_train = deproberta_features_train
    X_dev = deproberta_features_dev
    X_test = deproberta_features_test
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.01, 0.1, 1, 2, 5, 7, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
        'coef0': [0.0, 0.5, 1.0, 2.0, 3.0],
        'degree': [2, 3, 4]
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
    
    # Store the predictions on X_dev and X_test
    all_predictions_dev.append(y_pred_dev)
    all_predictions_test.append(y_pred_test)
    
    # Store the SVR model and its results for this prompt
    prompt_results = {
        'prompt_number': prompt_number,
        'model': svr,
        'train_rmse': np.sqrt(mse_train),
        'train_mae': mae_train,
        'dev_rmse': np.sqrt(mse_dev),
        'dev_mae': mae_dev,
        'test_rmse': np.sqrt(mse_test),
        'test_mae': mae_test
    }
    svr_models_results.append(prompt_results)

    # Calculate the weights based on 1/dev_mae
    weights = {prompt_results['prompt_number']: 1 / prompt_results['dev_mae'] for prompt_results in svr_models_results}
    print('weights: ', weights)

    # Save the SVR models and their results to files
    for prompt_results in svr_models_results:
        prompt_number = prompt_results['prompt_number']
        model_filename = f'svr_model_prompt_{prompt_number}.joblib'
        results_filename = f'svr_results_prompt_{prompt_number}.joblib'
        model = prompt_results['model']
        train_rmse = prompt_results['train_rmse']
        train_mae = prompt_results['train_mae']
        dev_rmse = prompt_results['dev_rmse']
        dev_mae = prompt_results['dev_mae']
        test_rmse = prompt_results['test_rmse']
        test_mae = prompt_results['test_mae']

        joblib.dump(model, model_filename)
        joblib.dump(
            {
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'dev_rmse': dev_rmse,
                'dev_mae': dev_mae,
                'test_rmse': test_rmse,
                'test_mae': test_mae
            },
            results_filename
        )

# ----------------------------------------------------------------------------------------
# Caculating the weighted mean for all feature sets

# Initialize a dictionary to store the weighted arrays
weighted_arrays_train = {}
weighted_arrays_dev = {}
weighted_arrays_test = {}

for prompt_features in all_features:
    prompt_number = prompt_features['prompt_number']
    print('prompt_number: ', prompt_number)

    X_train_features = np.array(prompt_features['X_train_features'])
    X_dev_features = np.array(prompt_features['X_dev_features'])
    X_test_features = np.array(prompt_features['X_test_features'])

    # Reshape the features
    X_train_features = X_train_features.reshape(X_train_features.shape[0], X_train_features.shape[2])
    X_dev_features = X_dev_features.reshape(X_dev_features.shape[0], X_dev_features.shape[2])
    X_test_features = X_test_features.reshape(X_test_features.shape[0], X_test_features.shape[2])

    # Weight calculation based on the 1/MAE(dev) for each pronpt base on the previous 
    # training results on the fine-tuned model for each of 10 prompts
    prompt_weight = weights.get(prompt_number)

    # Multiply the prompt features by the weight and store in the result array 
    # For train set
    weighted_prompt_features_train = X_train_features * prompt_weight
    # print('X_train_features: ', X_train_features)
    # print('weighted_prompt_features_train: ', weighted_prompt_features_train)
    print('weighted_prompt_features_train.shape: ', weighted_prompt_features_train.shape)

    # For dev set
    weighted_prompt_features_dev = X_dev_features * prompt_weight
    print('weighted_prompt_features_dev.shape: ', weighted_prompt_features_dev.shape)

    # For test set
    weighted_prompt_features_test = X_test_features * prompt_weight
    print('weighted_prompt_features_test.shape: ', weighted_prompt_features_test.shape)

    # Store the results in the dictionary with the prompt number as the key
    if prompt_number not in weighted_arrays_train:
        weighted_arrays_train[prompt_number] = []
    weighted_arrays_train[prompt_number].append(weighted_prompt_features_train)

    if prompt_number not in weighted_arrays_dev:
        weighted_arrays_dev[prompt_number] = []
    weighted_arrays_dev[prompt_number].append(weighted_prompt_features_dev)

    if prompt_number not in weighted_arrays_test:
        weighted_arrays_test[prompt_number] = []
    weighted_arrays_test[prompt_number].append(weighted_prompt_features_test)

# Convert the lists of arrays to numpy arrays
for prompt_number, arrays_list in weighted_arrays_train.items():
    weighted_arrays_train[prompt_number] = np.array(arrays_list)

for prompt_number, arrays_list in weighted_arrays_dev.items():
    weighted_arrays_dev[prompt_number] = np.array(arrays_list)

for prompt_number, arrays_list in weighted_arrays_test.items():
    weighted_arrays_test[prompt_number] = np.array(arrays_list)

# print('len(weighted_arrays_train): ', len(weighted_arrays_train))

# for prompt_number, array_list in weighted_arrays_train.items():
#     print(f"Prompt Number: {prompt_number}")
#     for index, weighted_array in enumerate(array_list):
#         print(f"Array {index + 1} shape:")
#         print(weighted_array.shape)

# Initialize the average_arrays for each set with zeros based on the shape (163, 3) and (56, 3) 
average_array_train_shape = (163, 3)
average_array_dev_shape = (56, 3)
average_array_test_shape = (56, 3)

average_array_train = np.zeros(average_array_train_shape)
average_array_dev = np.zeros(average_array_dev_shape)
average_array_test = np.zeros(average_array_test_shape)

# Iterate through each prompt's arrays
for prompt_arrays in weighted_arrays_train.values():
    for prompt_array in prompt_arrays:
        # print('prompt_array: ', prompt_array)
        average_array_train += prompt_array  # Add the current prompt's array to the running sum

for prompt_arrays in weighted_arrays_dev.values():
    for prompt_array in prompt_arrays:
        average_array_dev += prompt_array  # Add the current prompt's array to the running sum

for prompt_arrays in weighted_arrays_test.values():
    for prompt_array in prompt_arrays:
        average_array_test += prompt_array  # Add the current prompt's array to the running sum

# Divide the sum by the number of prompt arrays to get the average
average_array_train /= len(weighted_arrays_train)
average_array_dev /= len(weighted_arrays_dev)
average_array_test /= len(weighted_arrays_test)

# Now average_array contains the final average of all arrays
print('average_array_train.shape: ', average_array_train.shape)
print('average_array_dev.shape: ', average_array_dev.shape)
print('average_array_test.shape: ', average_array_test.shape)

# ------------------------------------------------------------------------------------------------
# Training and Evaluating and Testing using the features calculated baesd one the weighted average (1/MAE(dev))

# List to store SVR models and their results for each prompt
svr_models_results = []

# List to store predictions for each prompt
all_predictions_dev = []
all_predictions_test = []


X_train = average_array_train
X_dev = average_array_dev
X_test = average_array_test

y_train = np.array(df_train['PHQ_Score'])
print('final y_train shape: ', y_train.shape)
y_dev = np.array(df_dev['PHQ_Score'])
y_test = np.array(df_test['PHQ_Score'])

# Define the parameter grid for GridSearchCV
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.01, 0.1, 1, 2, 5, 7, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    'coef0': [0.0, 0.5, 1.0, 2.0, 3.0],
    'degree': [2, 3, 4]
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

# Predict on X_dev and calculate the mean squared error and mean absolute error
y_pred_dev = svr.predict(X_dev)
mse_dev = mean_squared_error(y_dev, y_pred_dev)
mae_dev = mean_absolute_error(y_dev, y_pred_dev)
print('RMSE for dev: ', np.sqrt(mse_dev))
print('MAE for dev: ', mae_dev)

# Predict on X_test and calculate the mean squared error and mean absolute error
y_pred_test = svr.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
print('RMSE for test: ', np.sqrt(mse_test))
print('MAE for test: ', mae_test)

print('y_dev: ', y_dev)
print('y_pred_dev: ', y_pred_dev)

print('y_test: ', y_test)
print('y_pred_test: ', y_pred_test)
# ----------------------------------------------------------------------------------------
# Calculate the ensemble prediction by taking the average of all predictions for each dev and test instance

# ensemble_prediction_dev = np.mean(all_predictions_dev, axis=0)
# ensemble_prediction_test = np.mean(all_predictions_test, axis=0)

# print('ensemble_prediction_dev: ', ensemble_prediction_dev)
# print('len ensemble_prediction_dev: ', len(ensemble_prediction_dev))

# # Calculate the mean squared error and mean absolute error for the ensemble prediction
# ensemble_mse_dev = mean_squared_error(y_dev, ensemble_prediction_dev)
# ensemble_mae_dev = mean_absolute_error(y_dev, ensemble_prediction_dev)
# print('Ensemble RMSE for dev: ', np.sqrt(ensemble_mse_dev))
# print('Ensemble MAE for dev: ', ensemble_mae_dev)

# # Calculate the mean squared error and mean absolute error for the ensemble prediction
# ensemble_mse_test = mean_squared_error(y_test, ensemble_prediction_test)
# ensemble_mae_test = mean_absolute_error(y_test, ensemble_prediction_test)
# print('Ensemble RMSE for test: ', np.sqrt(ensemble_mse_test))
# print('Ensemble MAE for test: ', ensemble_mae_test)
# # ------------------------------------------------------------------------------------------------
# # Implementing the Stacking Ensemble
# # The stacking ensemble involves training a meta-model that takes the predictions of 
# # multiple base models as input features and learns to make the final prediction. 
# # In this approach, you can split the data into multiple folds, and in each fold, 
# # train different SVR models on different training sets. Then, use the predictions 
# # from these models as input features to train the meta-model.

# # List to store predictions for each prompt
# all_predictions = []

# # Load the saved SVR models and their results
# for prompt_features in all_features:
#     prompt_number = prompt_features['prompt_number']
#     model_filename = f'svr_model_prompt_{prompt_number}.joblib'
#     model = joblib.load(model_filename)
    
#     X_test_features = np.array(prompt_features['X_test_features'])
#     deproberta_features_test = np.array(X_test_features)
#     deproberta_features_test = deproberta_features_test.reshape(deproberta_features_test.shape[0], deproberta_features_test.shape[2])
    
#     # Predict on X_test using the loaded SVR model
#     y_pred_test = model.predict(deproberta_features_test)
#     all_predictions.append(y_pred_test)

# # Convert the list of predictions into a numpy array
# meta_features = np.array(all_predictions)

# # Transpose the meta_features array to make it compatible for stacking
# meta_features = meta_features.T

# # Train the meta-model (SVR) on the meta_features and y_test
# meta_model = SVR(kernel='linear', C=1)  # You can choose a different SVR kernel and C value
# meta_model.fit(meta_features, y_test)

# # Predict on X_test using the stacking ensemble
# ensemble_prediction = meta_model.predict(meta_features)

# # Calculate the mean squared error and mean absolute error for the ensemble prediction
# ensemble_mse_test = mean_squared_error(y_test, ensemble_prediction)
# ensemble_mae_test = mean_absolute_error(y_test, ensemble_prediction)
# print('Ensemble (Stacking) RMSE for test: ', np.sqrt(ensemble_mse_test))
# print('Ensemble (Stacking) MAE for test: ', ensemble_mae_test)
